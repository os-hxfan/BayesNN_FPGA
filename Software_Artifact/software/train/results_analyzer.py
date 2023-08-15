import torch
import models
import train
import argparse
import types
import numpy as np
import datasets
import torch
import torch.nn.parallel
from torch import nn
from KDEpy import FFTKDE
from models.vgg19.vgg19 import VGG19EarlyExit, VGG19MC, VGG
from models.resnet18.resnet18 import ResNet18EarlyExit, ResNet18MC, ResNet
from train.hyperparameters import get_hyperparameters
from train.train_utils import get_device


class ResourceLoader():
    def __init__(self, gpu, dataset_name):
        # Specify Hyperparameters (maybe add command line compatibility?)
        self.hyperparameters = get_hyperparameters(self.fake_args(gpu, dataset_name))
        #self.hyperparameters["loaders"]["batch_size"] = (1,1,1)
        self.train_loader, self.val_loader, self.test_loader = datasets.get_dataloader(self.hyperparameters["loaders"])
        # Evaluate the Network on Test
        self.test_loss_fn = train.get_loss_function(self.hyperparameters["test_loss"])

    def get_model(self, model_num, model_type = "val", gpu = 0):
        if model_type == "val":
            path = "./MultiExit_BNNs/snapshots/"
            model_type = "best_val_model_"
        elif model_type == "test":
            path = "./MultiExit_BNNs/snapshots/"
            model_type = "final_model_"            
        print("Testing ", model_type+model_num)
        model_state = path+model_type+model_num
        self.hyperparameters["network"]["load_model"] = model_state
        self.hyperparameters["network"]["gpu_device"] = get_device(gpu)
        # Follow og for info on how to parallelize!
        model = models.get_network(self.hyperparameters["network"])
        return model
    
    def get_loader(self):
        return self.test_loader, self.val_loader
    
    def fake_args(self, gpu, dataset_name):
        args = types.SimpleNamespace(dropout_exit = False, dropout_p = 0.5,
            dropout_type = None, n_epochs = 300, patience = 50, backbone = "resnet", 
            single_exit = False, grad_clipping = 0, gpu = gpu, val_split = 0.1, reducelr_on_plateau = True,
            dataset_name = dataset_name, grad_accumulation = 0)
        return args

# Below class is a combination of methods from: https://github.com/yigitcankaya/Shallow-Deep-Networks/blob/1719a34163d55ff237467c542db86b7e1f9d7628/model_funcs.py
# and https://github.com/zhang64-llnl/Mix-n-Match-Calibration/blob/e41afbaf8181a0bd2fb194f9e9d30bcbe5b7f6c3/util_evaluation.py
# Both have been modified to work with Monte Carlo Dropout and Ensembling
class FullAnalysis():
    def __init__(self, model, test_loader, gpu=0, mc_dropout = False, mc_passes = 10, suffix = ""):
        self.model = model
        self.update_model_exits()
        self.loader = test_loader
        self.gpu = gpu
        self.mc_dropout = mc_dropout
        self.mc_passes = mc_passes
        self.sdn_get_detailed_results()
        self.filename_suffix = suffix

    def update_model_exits(self):
        if self.model.n_exits == 1 and (isinstance(self.model, VGG19EarlyExit) or isinstance(self.model, VGG19MC)):
            self.model.n_exits = 5
        elif self.model.n_exits == 1 and (isinstance(self.model, ResNet18EarlyExit) or isinstance(self.model, ResNet18MC)):
            self.model.n_exits = 4
        return None
            
    def multipass_experiment(self):
        passes = list(range(1,50))
        acc_list = []
        ensemble_acc_list = []
        ece_list = []
        ensemble_ece_list = []
        for i in range(len(passes)):
            self.mc_passes = passes[i]
            self.sdn_get_detailed_results()
            avg_acc, avg_ensembled_acc, avg_ece, avg_ensemble_ece = self.average_results_accuracy()
            acc_list.append(str(avg_acc))
            ensemble_acc_list.append(str(avg_ensembled_acc))
            ece_list.append(str(avg_ece))
            ensemble_ece_list.append(str(avg_ensemble_ece))
        print(",".join(acc_list))
        print(",".join(ensemble_acc_list))
        print(",".join(ece_list))
        print(",".join(ensemble_ece_list))
        self.mc_passes = 10
        return None

    def average_results_accuracy(self):
        acc = 0
        ensemble_acc = 0
        avg_ece = 0
        avg_ensemble_ece = 0
        for layer in range(len(self.layer_correct)):
            acc += len(self.layer_correct[layer])
            ensemble_acc += len(self.ensemble_layer_correct[layer])
            ece, _,_,_ = self.ece_eval_binary(self.preds[layer], self.labels)
            ensemble_ece, _,_,_ = self.ece_eval_binary(self.ensemble_preds[layer], self.labels)
            avg_ece += ece
            avg_ensemble_ece += ensemble_ece
        acc = acc/len(self.layer_correct)
        ensemble_acc = ensemble_acc/len(self.ensemble_layer_correct)
        avg_ece /= len(self.layer_correct)
        avg_ensemble_ece /= len(self.layer_correct)
        return acc, ensemble_acc, avg_ece, avg_ensemble_ece

    # https://github.com/yigitcankaya/Shallow-Deep-Networks/blob/1719a34163d55ff237467c542db86b7e1f9d7628/model_funcs.py#L156
    def sdn_get_detailed_results(self):
        """
        Returns:
        layer_correct : dict
            Each entry is a layer number containing the set of correctly
            classified instances 
        layer_wrong : dict
            Each entry is a layer number containing the set of incorrectly
            classified instances 
        layer_predictions : dict
            Each entry is a layer number containing a dictionary with the
            associated predictions for each instance
        layer_confidence : dict
            Each entry is a layer number containing a dictionary with the
            associated confidence of each prediction for each instance
        """
        self.device = get_device(self.gpu)
        self.model.eval()

        num_instances = len(self.loader.dataset)
        labels = np.zeros((num_instances,self.model.out_dim))
        ensembled_preds = np.empty((self.model.n_exits,num_instances,self.model.out_dim))
        preds = np.empty((self.model.n_exits,num_instances,self.model.out_dim))

        self.outputs = list(range(self.model.n_exits))
       
        layer_correct, layer_wrong, layer_predictions, layer_confidence = self._init_layer_trackers()
        ensemble_layer_correct, ensemble_layer_wrong, ensemble_layer_predictions, ensemble_layer_confidence = self._init_layer_trackers()

        with torch.no_grad():
            for cur_batch_id, batch in enumerate(self.loader):
                b_x = batch[0].to(self.device)
                b_y = batch[1].to(self.device, dtype=torch.long)
                output, output_sm, output_sm_np, ensemble_output, ensemble_output_sm = self._get_output(b_x)
                for output_id in self.outputs:
                    layer_correct, layer_wrong, layer_predictions, layer_confidence = self._update_layer_tracker(b_y, output_id, cur_batch_id, layer_correct, layer_wrong, layer_predictions, layer_confidence, output, output_sm)
                    ensemble_layer_correct, ensemble_layer_wrong, ensemble_layer_predictions, ensemble_layer_confidence = self._update_layer_tracker(b_y, output_id, cur_batch_id, ensemble_layer_correct, ensemble_layer_wrong, ensemble_layer_predictions, ensemble_layer_confidence, ensemble_output, ensemble_output_sm)
                for batch_instance_id in range(b_x.shape[0]):
                    if cur_batch_id == len(self.loader)-1:
                        instance_id = cur_batch_id*self.batch_size+batch_instance_id
                    else:
                        instance_id = cur_batch_id*b_x.shape[0]+batch_instance_id
                    correct_output = int(b_y[batch_instance_id].item())
                    labels[instance_id][correct_output] = 1
                # Will probably have issues if dataset size is not divisble by batch size
                if cur_batch_id == len(self.loader)-1:
                    preds[:,cur_batch_id*self.batch_size:b_x.shape[0]+cur_batch_id*self.batch_size, :] = output_sm_np
                else:
                    preds[:,cur_batch_id*b_x.shape[0]:(1+cur_batch_id)*b_x.shape[0], :] = output_sm_np
                self.batch_size = b_x.shape[0]
            for i in range(1,preds.shape[0]+1):
                ensembled_preds_per_layer = np.average(preds[:i,:,:],axis = 0)
                ensembled_preds[i-1,:,:] = ensembled_preds_per_layer
        self.layer_correct = layer_correct
        self.layer_wrong = layer_wrong
        self.layer_predictions = layer_predictions
        self.layer_confidence = layer_confidence
        self.ensemble_layer_correct = ensemble_layer_correct
        self.ensemble_layer_wrong = ensemble_layer_wrong
        self.ensemble_layer_predictions = ensemble_layer_predictions
        self.ensemble_layer_confidence = ensemble_layer_confidence
        self.preds = preds
        self.ensemble_preds = ensembled_preds
        self.labels = labels
        return None

    def get_validation_predictions(self, val_loader):
        self.device = get_device(self.gpu)
        self.model.eval()
        if isinstance(val_loader.sampler,torch.utils.data.sampler.SubsetRandomSampler):
            num_instances = len(val_loader.sampler.indices)
        else:
            num_instances = len(self.loader.dataset)
        layer_correct, layer_wrong, layer_predictions, layer_confidence = self._init_layer_trackers()
        ensemble_layer_correct, ensemble_layer_wrong, ensemble_layer_predictions, ensemble_layer_confidence = self._init_layer_trackers()
        labels = np.zeros((num_instances,self.model.out_dim))
        ensembled_preds = np.empty((self.model.n_exits,num_instances,self.model.out_dim))
        preds = np.empty((self.model.n_exits,num_instances,self.model.out_dim))
        self.outputs = list(range(self.model.n_exits))
        with torch.no_grad():
            for cur_batch_id, batch in enumerate(val_loader):
                b_x = batch[0].to(self.device)
                b_y = batch[1].to(self.device, dtype=torch.long)
                output, output_sm, output_sm_np, ensemble_output, ensemble_output_sm = self._get_output(b_x)
                for output_id in self.outputs:
                    layer_correct, layer_wrong, layer_predictions, layer_confidence = self._update_layer_tracker(b_y, output_id, cur_batch_id, layer_correct, layer_wrong, layer_predictions, layer_confidence, output, output_sm)
                    ensemble_layer_correct, ensemble_layer_wrong, ensemble_layer_predictions, ensemble_layer_confidence = self._update_layer_tracker(b_y, output_id, cur_batch_id, ensemble_layer_correct, ensemble_layer_wrong, ensemble_layer_predictions, ensemble_layer_confidence, ensemble_output, ensemble_output_sm)
                for batch_instance_id in range(b_x.shape[0]):
                    if cur_batch_id == len(val_loader)-1:
                        instance_id = cur_batch_id*self.batch_size+batch_instance_id
                    else:
                        instance_id = cur_batch_id*b_x.shape[0]+batch_instance_id
                    correct_output = int(b_y[batch_instance_id].item())
                    labels[instance_id][correct_output] = 1
                if cur_batch_id == len(val_loader)-1:
                    preds[:,cur_batch_id*self.batch_size:b_x.shape[0]+cur_batch_id*self.batch_size, :] = output_sm_np
                else:
                    preds[:,cur_batch_id*b_x.shape[0]:(1+cur_batch_id)*b_x.shape[0], :] = output_sm_np
                self.batch_size = b_x.shape[0]
            for i in range(1,preds.shape[0]+1):
                ensembled_preds_per_layer = np.average(preds[:i,:,:],axis = 0)
                ensembled_preds[i-1,:,:] = ensembled_preds_per_layer
        return preds, ensembled_preds, labels

    def save_validation(self, experiment_id, loader):
        preds, ensemble_preds, labels = self.get_validation_predictions(loader)
        with open(f"validation_predictions_{experiment_id}.npy", 'wb') as file:
            np.save(file, preds)
            np.save(file, ensemble_preds)
            np.save(file, labels)

    def _init_layer_trackers(self):
        layer_correct = {}
        layer_wrong = {}
        layer_predictions = {}
        layer_confidence = {}
        for output_id in self.outputs:
            layer_correct[output_id] = set()
            layer_wrong[output_id] = set()
            layer_predictions[output_id] = {}
            layer_confidence[output_id] = {}
        return layer_correct, layer_wrong, layer_predictions, layer_confidence

    def _get_output(self, b_x):
        if self.mc_dropout:
            all_output_probs = np.empty((self.mc_passes,len(self.outputs),b_x.shape[0],self.model.out_dim))
            all_outputs = np.empty((self.mc_passes,len(self.outputs),b_x.shape[0],self.model.out_dim))
            for i in range(self.mc_passes):
                output = self.model(b_x)
                output_sm = [nn.functional.softmax(out, dim=1).cpu().numpy() for out in output]
                output = [out.cpu().numpy() for out in output]
                output_sm_np = np.asarray(output_sm)
                all_outputs[i,:] = np.asarray(output)
                all_output_probs[i,:] = output_sm_np
            output_np = np.average(all_outputs,axis = 0)
            output_sm_np = np.average(all_output_probs,axis = 0)
            output_sm = []
            output = []
            for i in range(output_sm_np.shape[0]):
                output_sm.append(torch.from_numpy(output_sm_np[i]))
                output.append(torch.from_numpy(output_np[i]))
        else:
            output = self.model(b_x)
            output_sm = [nn.functional.softmax(out, dim=1) for out in output]
            output_sm_np = [nn.functional.softmax(out, dim=1).cpu().numpy() for out in output]
            output_sm_np = np.asarray(output_sm_np)

        ensemble_output_sm = []
        ensemble_output = []
        for i in range(len(output_sm)):
            ensemble_i_output_probs = output_sm[:i+1]
            ensemble_i_outputs = output[:i+1]
            for array in range(len(ensemble_i_output_probs)):
                ensemble_i_output_probs[array] = ensemble_i_output_probs[array].cpu().numpy()
                ensemble_i_outputs[array] = ensemble_i_outputs[array].cpu().numpy()
            ensemble_output_sm.append(torch.from_numpy(np.mean(ensemble_i_output_probs,axis = 0)))
            ensemble_output.append(torch.from_numpy(np.mean(ensemble_i_outputs,axis = 0)))
        return output, output_sm, output_sm_np, ensemble_output, ensemble_output_sm

    def _update_layer_tracker(self, b_y, output_id, cur_batch_id, layer_correct, layer_wrong, layer_predictions, layer_confidence, output, output_sm):
        cur_output = output[output_id]
        cur_confidences = output_sm[output_id].max(1, keepdim=True)[0]
        pred = cur_output.max(1, keepdim=True)[1].to(self.device)
        is_correct = pred.eq(b_y.view_as(pred))
        for test_id in range(len(b_y)):
            cur_instance_id = test_id + cur_batch_id*self.loader.batch_size
            correct = is_correct[test_id]
            layer_predictions[output_id][cur_instance_id] = pred[test_id].cpu().numpy()
            layer_confidence[output_id][cur_instance_id] = cur_confidences[test_id].cpu().numpy()
            if correct == 1:
                layer_correct[output_id].add(cur_instance_id)
            else:
                layer_wrong[output_id].add(cur_instance_id) 
        return layer_correct, layer_wrong, layer_predictions, layer_confidence

    def all_experiments(self, experiment_id):
        layers = sorted(list(self.layer_correct.keys()))
        end_wrong = self.layer_wrong[layers[-1]]
        ensemble_end_wrong = self.ensemble_layer_wrong[layers[-1]]
        cum_correct = set()
        ensemble_cum_correct = set()
        self.cur_correct_saver = []
        self.cum_correct_saver = []
        self.destructive_overthinking = []
        self.unique_correct_saver = []
        self.ece_saver = []
        self.nll_saver = []
        self.mse_saver = []
        self.accu_saver = []
        self.ensemble_cur_correct_saver = []
        self.ensemble_cum_correct_saver = []
        self.ensemble_destructive_overthinking = []
        self.ensemble_unique_correct = []
        self.ensemble_ece_saver = []
        self.ensemble_nll_saver = []
        self.ensemble_mse_saver = []
        self.ensemble_accu_saver = []
        for layer in layers:
            cur_correct = self.layer_correct[layer]
            unique_correct = cur_correct - cum_correct
            cum_correct = cum_correct | cur_correct
            cur_overthinking = cur_correct & end_wrong
            self.cur_correct_saver.append(len(cur_correct))
            self.cum_correct_saver.append(len(cum_correct))
            self.destructive_overthinking.append(len(cur_overthinking))
            self.unique_correct_saver.append(len(unique_correct))
            ece, nll, mse, accu = self.ece_eval_binary(self.preds[layer], self.labels)
            self.ece_saver.append(ece)
            self.nll_saver.append(nll)
            self.mse_saver.append(mse)
            self.accu_saver.append(accu)
            cur_correct = self.ensemble_layer_correct[layer]
            unique_correct = cur_correct - ensemble_cum_correct
            ensemble_cum_correct = ensemble_cum_correct | cur_correct
            cur_overthinking = cur_correct & ensemble_end_wrong
            self.ensemble_cur_correct_saver.append(len(cur_correct))
            self.ensemble_cum_correct_saver.append(len(ensemble_cum_correct))
            self.ensemble_destructive_overthinking.append(len(cur_overthinking))
            self.ensemble_unique_correct.append(len(unique_correct))
            ece_ensemble, nll_ensemble, mse_ensemble, accu_ensemble = self.ece_eval_binary(self.ensemble_preds[layer], self.labels)
            self.ensemble_ece_saver.append(ece_ensemble)
            self.ensemble_nll_saver.append(nll_ensemble)
            self.ensemble_mse_saver.append(mse_ensemble)
            self.ensemble_accu_saver.append(accu_ensemble)
        self.saver(experiment_id)

    def mirror_1d(self, d, xmin=None, xmax=None):
        """If necessary apply reflecting boundary conditions."""
        if xmin is not None and xmax is not None:
            xmed = (xmin+xmax)/2
            return np.concatenate(((2*xmin-d[d < xmed]).reshape(-1,1), d, (2*xmax-d[d >= xmed]).reshape(-1,1)))
        elif xmin is not None:
            return np.concatenate((2*xmin-d, d))
        elif xmax is not None:
            return np.concatenate((d, 2*xmax-d))
        else:
            return d

    def ece_kde_binary(self, p,label,p_int=None,order=1):

        # points from numerical integration
        if p_int is None:
            p_int = np.copy(p)

        p = np.clip(p,1e-256,1-1e-256)
        p_int = np.clip(p_int,1e-256,1-1e-256)
        
        
        x_int = np.linspace(-0.6, 1.6, num=2**14)
        
        
        N = p.shape[0]

        # this is needed to convert labels from one-hot to conventional form
        label_index = np.array([np.where(r==1)[0][0] for r in label])
        with torch.no_grad():
            if p.shape[1] !=2:
                p_new = torch.from_numpy(p)
                p_b = torch.zeros(N,1)
                label_binary = np.zeros((N,1))
                for i in range(N):
                    pred_label = int(torch.argmax(p_new[i]).numpy())
                    if pred_label == label_index[i]:
                        label_binary[i] = 1
                    p_b[i] = p_new[i,pred_label]/torch.sum(p_new[i,:])  
            else:
                p_b = torch.from_numpy((p/np.sum(p,1)[:,None])[:,1])
                label_binary = label_index
                    
        method = 'triweight'
        dconf_1 = (p_b[np.where(label_binary==1)].reshape(-1,1)).numpy()
        if np.std(dconf_1) != 0:
            kbw = np.std(p_b.numpy())*(N*2)**-0.2
            kbw = np.std(dconf_1)*(N*2)**-0.2
        else: # Numerical stability
            kbw = 0.0000000000000001*(N*2)**-0.2

        # Mirror the data about the domain boundary
        low_bound = 0.0
        up_bound = 1.0
        dconf_1m = self.mirror_1d(dconf_1,low_bound,up_bound)
        # Compute KDE using the bandwidth found, and twice as many grid points
        pp1 = FFTKDE(bw=kbw, kernel=method).fit(dconf_1m).evaluate(x_int)
        pp1[x_int<=low_bound] = 0  # Set the KDE to zero outside of the domain
        pp1[x_int>=up_bound] = 0  # Set the KDE to zero outside of the domain
        pp1 = pp1 * 2  # Double the y-values to get integral of ~1
        
        
        p_int = p_int/np.sum(p_int,1)[:,None]
        N1 = p_int.shape[0]
        with torch.no_grad():
            p_new = torch.from_numpy(p_int)
            pred_b_int = np.zeros((N1,1))
            if p_int.shape[1]!=2:
                for i in range(N1):
                    pred_label = int(torch.argmax(p_new[i]).numpy())
                    pred_b_int[i] = p_int[i,pred_label]
            else:
                for i in range(N1):
                    pred_b_int[i] = p_int[i,1]

        low_bound = 0.0
        up_bound = 1.0
        pred_b_intm = self.mirror_1d(pred_b_int,low_bound,up_bound)
        # Compute KDE using the bandwidth found, and twice as many grid points
        pp2 = FFTKDE(bw=kbw, kernel=method).fit(pred_b_intm).evaluate(x_int)
        pp2[x_int<=low_bound] = 0  # Set the KDE to zero outside of the domain
        pp2[x_int>=up_bound] = 0  # Set the KDE to zero outside of the domain
        pp2 = pp2 * 2  # Double the y-values to get integral of ~1

        
        if p.shape[1] !=2: # top label (confidence)
            perc = np.mean(label_binary)
        else: # or joint calibration for binary cases
            perc = np.mean(label_index)
                
        integral = np.zeros(x_int.shape)
        reliability= np.zeros(x_int.shape)
        for i in range(x_int.shape[0]):
            conf = x_int[i]
            if np.max([pp1[np.abs(x_int-conf).argmin()],pp2[np.abs(x_int-conf).argmin()]])>1e-6:
                accu = np.min([perc*pp1[np.abs(x_int-conf).argmin()]/pp2[np.abs(x_int-conf).argmin()],1.0])
                if np.isnan(accu)==False:
                    integral[i] = np.abs(conf-accu)**order*pp2[i]  
                    reliability[i] = accu
            else:
                if i>1:
                    integral[i] = integral[i-1]

        ind = np.where((x_int >= 0.0) & (x_int <= 1.0))
        return np.trapz(integral[ind],x_int[ind])/np.trapz(pp2[ind],x_int[ind])


    def ece_hist_binary(self,p, label, n_bins = 15, order=1):
        
        p = np.clip(p,1e-256,1-1e-256)
        
        N = p.shape[0]
        label_index = np.array([np.where(r==1)[0][0] for r in label]) # one hot to index
        with torch.no_grad():
            if p.shape[1] !=2:
                preds_new = torch.from_numpy(p)
                preds_b = torch.zeros(N,1)
                label_binary = np.zeros((N,1))
                for i in range(N):
                    pred_label = int(torch.argmax(preds_new[i]).numpy())
                    if pred_label == label_index[i]:
                        label_binary[i] = 1
                    preds_b[i] = preds_new[i,pred_label]/torch.sum(preds_new[i,:])  
            else:
                preds_b = torch.from_numpy((p/np.sum(p,1)[:,None])[:,1])
                label_binary = label_index

            confidences = preds_b
            accuracies = torch.from_numpy(label_binary)


            x = confidences.numpy()
            x = np.sort(x,axis=0)
            binCount = int(len(x)/n_bins) #number of data points in each bin
            bins = np.zeros(n_bins) #initialize the bins values
            for i in range(0, n_bins, 1):
                bins[i] = x[min((i+1) * binCount,x.shape[0]-1)]
                #print((i+1) * binCount)
            bin_boundaries = torch.zeros(len(bins)+1,1)
            bin_boundaries[1:] = torch.from_numpy(bins).reshape(-1,1)
            bin_boundaries[0] = 0.0
            bin_boundaries[-1] = 1.0
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            
            ece_avg = torch.zeros(1)
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                # Calculated |confidence - accuracy| in each bin
                in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
                prop_in_bin = in_bin.float().mean()
                #print(prop_in_bin)
                if prop_in_bin.item() > 0:
                    accuracy_in_bin = accuracies[in_bin].float().mean()
                    avg_confidence_in_bin = confidences[in_bin].mean()
                    ece_avg += torch.abs(avg_confidence_in_bin - accuracy_in_bin)**order * prop_in_bin
        return ece_avg

    def ece_eval_binary(self,p, label):
        mse = np.mean(np.sum((p-label)**2,1)) # Mean Square Error
        N = p.shape[0]
        p = np.clip(p,1e-256,1-1e-256)
        nll = -np.sum(label*np.log(p))/N # log_likelihood
        accu = (np.sum((np.argmax(p,1)-np.array([np.where(r==1)[0][0] for r in label]))==0)/p.shape[0]) # Accuracy
        ece = self.ece_kde_binary(p,label)   

        return ece, nll, mse, accu


    def saver(self, experiment_id):
        # Print in correct format for processing
        self.model_type = self.get_model_type()
        self.exit_only, dropout_rate, mc_passes = self.get_dropout_type()
        self.get_flops_per_module()
        # Use to save results in a useful manner
        # Note: Layer is either 0,1,2... or Ensemble 1, Ensemble 2...
        #Layer,Accuracy,Cumulative Correct,Destructive Overthinking,Unique Correct,ECE,NLL,MSE
        with open("test_evaluation_log_"+self.model_type+str(experiment_id)+self.filename_suffix+".txt", "w") as file:
            for layer in range(len(self.cur_correct_saver)):
                list_str = [str(layer),str(self.accu_saver[layer]), str(self.cum_correct_saver[layer]), str(self.destructive_overthinking[layer]),
                    str(self.unique_correct_saver[layer]), str(self.ece_saver[layer]), str(self.nll_saver[layer]),str(self.mse_saver[layer])]
                full_str = ",".join(list_str) + "\n"
                file.write(full_str)
            for layer in range(len(self.cur_correct_saver)):
                list_str = ["Ensemble" + str(layer),str(self.ensemble_accu_saver[layer]), str(self.ensemble_cum_correct_saver[layer]), str(self.ensemble_destructive_overthinking[layer]),
                    str(self.ensemble_unique_correct[layer]), str(self.ensemble_ece_saver[layer]), str(self.ensemble_nll_saver[layer]),str(self.ensemble_mse_saver[layer])]
                full_str = ",".join(list_str) + "\n"
                file.write(full_str)
                # Print for Ease of Access
                if self.exit_only:
                    print(f"E ({dropout_rate},{str(layer)}), {str(self.accu_saver[layer])}, {str(self.ece_saver[layer])}, {str(self.get_flops_standard_exit(layer, mc_passes, ensemble = False)/self.baseline_flops)}, {self.nll_saver[layer]}")
                    print(f"E ({dropout_rate},Ensemble{str(layer)}), {str(self.ensemble_accu_saver[layer])}, {str(self.ensemble_ece_saver[layer])}, {str(self.get_flops_standard_exit(layer, mc_passes, ensemble = True)/self.baseline_flops)}, {self.ensemble_nll_saver[layer]}")
                elif self.model.dropout == "block":
                    print(f"B+E ({dropout_rate},{str(layer)}), {str(self.accu_saver[layer])}, {str(self.ece_saver[layer])}, {self.nll_saver[layer]} ")
                    print(f"B+E ({dropout_rate},Ensemble{str(layer)}), {str(self.ensemble_accu_saver[layer])}, {str(self.ensemble_ece_saver[layer])}, {self.ensemble_nll_saver[layer]}, ")
                elif self.model.dropout == "layer":  
                    print(f"L+E ({dropout_rate},{str(layer)}), {str(self.accu_saver[layer])}, {str(self.ece_saver[layer])}, {self.nll_saver[layer]} ")
                    print(f"L+E ({dropout_rate},Ensemble{str(layer)}), {str(self.ensemble_accu_saver[layer])}, {str(self.ensemble_ece_saver[layer])}, {self.ensemble_nll_saver[layer]}, ")                                  
        # Save ensemble, preds and labels for further analysis
        with open(f"test_predictions_{experiment_id}.npy", 'wb') as file:
            np.save(file, self.preds)
            np.save(file, self.ensemble_preds)
            np.save(file, self.labels)

    def get_confidence_exiting_values(self, model_num):
        self.model_type = self.get_model_type()
        self.exit_only, dropout_rate, mc_passes = self.get_dropout_type()
        self.get_flops_per_module()
        with open(f"test_predictions_{model_num}.npy", 'rb') as file:
            p_evals = np.load(file)
            ensembled_p_evals = np.load(file)
            labels = np.load(file)
        confidence_list = [0.1, 0.15, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999]
        for threshold in confidence_list:
            ensembled_flops = self.flop_saver_ensembled(threshold, ensembled_p_evals, labels, mc_passes = mc_passes, diff = False)
            flops = self.flop_saver(threshold, p_evals, labels, mc_passes = mc_passes, diff = False)
            ensembled_accuracy, ensembled_ece, ensembled_nll = self.confidence_exiting(threshold, ensembled_p_evals, labels, diff = False)
            accuracy, ece, nll = self.confidence_exiting(threshold, p_evals, labels, diff = False)
            if self.exit_only:
                print(f"E ({dropout_rate},{threshold}), {accuracy}, {ece}, {flops/(self.baseline_flops*10000)}, {nll}")
                print(f"Ensemble E ({dropout_rate},{threshold}), {ensembled_accuracy}, {ensembled_ece}, {ensembled_flops/(self.baseline_flops*10000)}, {ensembled_nll}")
            elif self.model.dropout == "block":
                print(f"B+E ({dropout_rate},{threshold}), {accuracy}, {ece}, ,{flops}, {nll}")
                print(f"Ensemble B+E ({dropout_rate},{threshold}), {ensembled_accuracy}, {ensembled_ece}, ,{ensembled_flops}, {ensembled_nll}")
            elif self.model.dropout == "layer":
                print(f"L+E ({dropout_rate},{threshold}), {accuracy}, {ece}, ,{flops}, {nll}")
                print(f"Ensemble L+E ({dropout_rate},{threshold}), {ensembled_accuracy}, {ensembled_ece}, ,{ensembled_flops}, {ensembled_nll}")
        return None
    
    def get_flops_per_module(self):
        if self.model_type == "vgg19":
            self.n_exits = 5
            self.flops_per_layer = [40173568,	56950784,	132448256,	132284416,	37789696]
            self.flop_per_exit_convs = [14227456,	9467904,	4728832,	0,	0]
            self.flops_per_exit = [51200,	51200,	51200,	51200,	51200]
        elif self.model_type == "resnet18":
            self.n_exits = 4
            self.flops_per_layer =[154402816,	135036928,	134627328,	134422528]
            self.flop_per_exit_convs =[56909824,	37871616,	18915328, 0]
            self.flops_per_exit =[51200,	51200,	51200, 51200]
        self.baseline_flops = sum(self.flops_per_layer) + self.flop_per_exit_convs[-1] + self.flops_per_exit[-1]
        return None

    def get_dropout_type(self):
        try:
            if self.model.dropout_exit and self.model.dropout is None:
                exit_only = True
            elif self.model.dropout is not None:
                exit_only = False
            mc_passes = 10
            dropout_rate = self.model.dropout_p
        except:
            # This bool matters! (Prevents a thrown if statement later)
            exit_only = True
            dropout_rate = 0
            mc_passes = 1
        return exit_only, dropout_rate, mc_passes


    def get_model_type(self):
        if isinstance(self.model,VGG):
            return "vgg19"
        elif isinstance(self.model,ResNet):
            return "resnet18"
        else:
            raise ValueError

    def confidence_exiting(self, threshold, p_evals, labels, diff = False):
        instances = set(range(labels.shape[0]))
        correct = set()
        incorrect = set()
        seen = set()
        best_preds = np.zeros((labels.shape[0],labels.shape[1]))
        for layer in range(1,self.n_exits):
            for instance in instances:
                if self.is_confident(p_evals, threshold, layer, instance, diff = diff):
                    seen.add(instance)
                    best_preds[instance,:] = p_evals[layer][instance]
                    if np.argmax(labels[instance]).item() == np.argmax(p_evals[layer][instance]).item():
                        correct.add(instance)
                    else:
                        incorrect.add(instance)
                elif layer == self.n_exits-1:
                    seen.add(instance)
                    best_preds[instance,:] = p_evals[layer][instance]
                    if np.argmax(labels[instance]).item() == np.argmax(p_evals[layer][instance]).item():
                        correct.add(instance)
                    else:
                        incorrect.add(instance)
            instances = instances.difference(seen)
        ece, nll, mse, accu = self.ece_eval_binary(best_preds,labels)
        return accu, ece, nll

    def get_flops_standard_exit(self, layer, mc_passes, ensemble = False):
        if ensemble:
            flops = sum(self.flops_per_layer[:layer+1])+sum(self.flop_per_exit_convs[:layer+1])+sum(self.flops_per_exit[:layer+1])*mc_passes
        else:
            flops = sum(self.flops_per_layer[:layer+1])+self.flop_per_exit_convs[layer]+self.flops_per_exit[layer]*mc_passes
        return flops

    def flop_saver(self,threshold, p_evals, labels, mc_passes = 10, diff = False):
        instances = set(range(labels.shape[0]))
        correct = set()
        incorrect = set()
        seen = set()
        flops_total = 0
        for layer in range(1,self.n_exits):
            for instance in instances:
                if layer == self.n_exits - 1:
                    if self.model_type == "vgg19" or self.model_type == "resnet18":
                        block_flops = sum(self.flops_per_layer[:layer+1])
                    else:
                        block_flops = sum(self.flops_per_layer[:layer])
                    if self.exit_only:
                        flops_total += block_flops + self.flop_per_exit_convs[layer] + mc_passes*self.flops_per_exit[layer]
                    else:
                        flops_total += mc_passes*(block_flops + self.flop_per_exit_convs[layer] + self.flops_per_exit[layer])
                    seen.add(instance)
                    if np.argmax(labels[instance]).item() == np.argmax(p_evals[layer][instance]).item():
                        correct.add(instance)
                    else:
                        incorrect.add(instance)
                elif self.is_confident(p_evals, threshold, layer, instance, diff = diff):
                    if self.exit_only:
                        flops_total += sum(self.flops_per_layer[:layer+1]) + self.flop_per_exit_convs[layer] + mc_passes*self.flops_per_exit[layer]
                    else:
                        flops_total += mc_passes*(sum(self.flops_per_layer[:layer+1]) + self.flop_per_exit_convs[layer] + self.flops_per_exit[layer])
                    seen.add(instance)
                    if np.argmax(labels[instance]).item() == np.argmax(p_evals[layer][instance]).item():
                        correct.add(instance)
                    else:
                        incorrect.add(instance)
            instances = instances.difference(seen)
        return flops_total

    def flop_saver_ensembled(self,threshold, p_evals, labels, mc_passes = 10, diff = False):
        instances = set(range(labels.shape[0]))
        correct = set()
        incorrect = set()
        seen = set()
        flops_total = 0
        counter = 0
        for layer in range(1,self.n_exits):
            for instance in instances:
                if layer == self.n_exits - 1:
                    counter += 1
                    if self.model_type == "vgg19" or self.model_type == "resnet18":
                        flop_instance = sum(self.flops_per_layer[:layer+1])
                    else:
                        flop_instance = sum(self.flops_per_layer[:layer])
                    if self.exit_only:
                        for prev_layer in range(layer+1):
                            flop_instance += self.flop_per_exit_convs[prev_layer]
                            flop_instance += mc_passes*self.flops_per_exit[prev_layer]
                    else:
                        for prev_layer in range(layer+1):
                            flop_instance += self.flop_per_exit_convs[prev_layer]
                            flop_instance += self.flops_per_exit[prev_layer]
                        flop_instance *= mc_passes
                    flops_total += flop_instance
                    seen.add(instance)
                    if np.argmax(labels[instance]).item() == np.argmax(p_evals[layer][instance]).item():
                        correct.add(instance)
                    else:
                        incorrect.add(instance)

                elif self.is_confident(p_evals, threshold, layer, instance, diff = diff):
                    flop_instance = sum(self.flops_per_layer[:layer+1])
                    if self.exit_only:
                        for prev_layer in range(layer+1):
                            flop_instance += self.flop_per_exit_convs[prev_layer]
                            flop_instance += mc_passes*self.flops_per_exit[prev_layer]
                    else:
                        for prev_layer in range(layer+1):
                            flop_instance += self.flop_per_exit_convs[prev_layer]
                            flop_instance += self.flops_per_exit[prev_layer]
                        flop_instance *= mc_passes
                    flops_total += flop_instance

                    seen.add(instance)
                    if np.argmax(labels[instance]).item() == np.argmax(p_evals[layer][instance]).item():
                        correct.add(instance)
                    else:
                        incorrect.add(instance)

            instances = instances.difference(seen)
        return flops_total


    def is_confident(self,p_evals, threshold, layer, instance, diff = False):
        if diff:
            temp = np.partition(-p_evals[layer][instance], 2)
            result = -temp[:2]
            return abs(result[0]-result[1]) > threshold
        else:
            return np.max(p_evals[layer][instance]).item() > threshold

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adding dropout")
    parser.add_argument('--model_num', type=str, default='75')
    parser.add_argument('--model_type', type=str, default='val')
    parser.add_argument('--dropout', type = bool, default = False)
    parser.add_argument('--num_passes', type=int, default=10)
    parser.add_argument('--confidence_exiting', type=bool, default = False)
    parser.add_argument('--overthinking', type=bool,default = False)
    parser.add_argument('--uq', type=bool, default = False)
    parser.add_argument('--plotting',type=bool,default = False)
    parser.add_argument('--ensemble', type=bool, default = False)
    parser.add_argument('--testing', type=bool, default = False)
    parser.add_argument('--full_and_save', type=bool, default = False)
    parser.add_argument('--multiple_pass', type=bool, default = False)
    parser.add_argument('--gpu', type=int, default = 0)
    parser.add_argument('--automated_experiment', type=bool, default = False)
    parser.add_argument('--dataset_name', type=str, default = "chestx")
    args = parser.parse_args()
    resources = ResourceLoader(args.gpu, args.dataset_name)
    test_loader, val_loader = resources.get_loader()
    model = resources.get_model(args.model_num, model_type = "val", gpu = args.gpu)
    if args.full_and_save:
        full_analyzer = FullAnalysis(model, test_loader, gpu = args.gpu, 
            mc_dropout = args.dropout, mc_passes = args.num_passes)
        full_analyzer.all_experiments(args.model_num)
        full_analyzer.save_validation(args.model_num, val_loader)
        full_analyzer.get_confidence_exiting_values(args.model_num)

    if args.multiple_pass:
        full_analyzer = FullAnalysis(model, test_loader, gpu = args.gpu, 
            mc_dropout = args.dropout, mc_passes = args.num_passes)  
        full_analyzer.multipass_experiment()
    
    if args.automated_experiment:
        # Non Dropout
        model_list = [16, 338]
        for model_num in model_list:
            model_num = str(model_num)
            model = resources.get_model(model_num, model_type = "val", gpu = args.gpu)
            full_analyzer = FullAnalysis(model, test_loader, gpu = args.gpu, 
                mc_dropout = False, mc_passes = 1)
            full_analyzer.all_experiments(model_num)
            #full_analyzer.save_validation(model_num, val_loader)
            full_analyzer.get_confidence_exiting_values(model_num)
        # Dropout
        model_list = [337, 339, 342, 340, 91, 18, 19, 17]
        for model_num in model_list:
            model_num = str(model_num)
            model = resources.get_model(model_num, model_type = "val", gpu = args.gpu)
            full_analyzer = FullAnalysis(model, test_loader, gpu = args.gpu, 
                mc_dropout = True, mc_passes = 10)
            full_analyzer.all_experiments(model_num)
            #full_analyzer.save_validation(model_num, val_loader)
            full_analyzer.get_confidence_exiting_values(model_num)