import xmltodict
import sys 
import os 
import matplotlib.pyplot as plt

def parse_xml(file_name):
  with open(file_name)as fd:
    d = xmltodict.parse(fd.read()) 
  return d 

# iterate over a give dictionary, and parse all xml files to dicitonary. Then extract the latency information.
def parse_stats_from_xmls(folder, prefix, 
                          path=['profile', 
                                'PerformanceEstimates',
                                'SummaryOfOverallLatency', 
                                'Average-caseRealTimeLatency']):
  ls = []
  files = []
  for file in os.listdir(folder):
    if file.endswith('.xml') and file.startswith(prefix):
      files.append(os.path.join(folder, file))
  files.sort()
  for file in files:
      d = parse_xml(file)
      for p in path:
        d = d[p]
      s = d.split(' ')
      latency = float(s[0])
      if len(s) > 1 and s[1] == 'sec':
        latency = latency * 1000
      print(file, latency)
      ls.append(latency)
  return ls

def parse_latency_from_xmls(folder, prefix):
  return parse_stats_from_xmls(folder, prefix, path=['profile', 
                                'PerformanceEstimates',
                                'SummaryOfOverallLatency', 
                                'Average-caseRealTimeLatency'])

def parse_LUT_from_xmls(folder, prefix):
  return parse_stats_from_xmls(folder, prefix, path=['profile', 
                                'AreaEstimates',
                                'Resources', 
                                'LUT'])

def parse_BRAM_18K_from_xmls(folder, prefix):
  return parse_stats_from_xmls(folder, prefix, path=['profile', 
                                'AreaEstimates',
                                'Resources', 
                                'BRAM_18K'])

def parse_FF_from_xmls(folder, prefix):
  return parse_stats_from_xmls(folder, prefix, path=['profile', 
                                'AreaEstimates',
                                'Resources', 
                                'FF'])

def draw_graph(f, ylabel):
  masksembles_resnet = []
  mc_dropouts_resnet = []
  masksembles_vgg = []
  mc_dropouts_vgg = []
  masksembles_lenet = []
  mc_dropouts_lenet = []
  for net in ['ResNet18', 'LeNet', 'VGG11']:
    for folder in ['../../autobayes/diff_dropouts', '../../autobayes/diff_masksembles']: 
      d = f(folder, net)
      if folder == 'extension/diff_dropouts':
        if net == 'ResNet18':
          mc_dropouts_resnet = d
        elif net == 'VGG11': 
          mc_dropouts_vgg = d
        else:
          mc_dropouts_lenet = d
      else: 
        if net == 'ResNet18':
          masksembles_resnet = d
        elif net == 'VGG11':
          masksembles_vgg = d
        else:
          masksembles_lenet = d
  print('mc_dropouts_resnet', mc_dropouts_resnet)
  print('masksembles_resnet', masksembles_resnet)
  print('mc_dropouts_lenet', mc_dropouts_lenet)
  print('masksembles_lenet', masksembles_lenet) 
  print('mc_dropouts_vgg', mc_dropouts_vgg)
  print('masksembles_vgg', masksembles_vgg)

  x_lenet = [i for i in range(4)] 
  x_resnet = [i for i in range(9)] 
  x_vgg = [i for i in range(9)]
  _, ax = plt.subplots(3, figsize=(10, 15))

  ax[0].plot(x_lenet, mc_dropouts_lenet, label='mc_dropout')
  ax[0].plot(x_lenet, masksembles_lenet, label='masksembles')
  ax[0].set_xlabel("number of dropout/masksembles layers")
  ax[0].set_ylabel(ylabel)
  ax[0].legend()
  ax[0].set_title('LeNet') 

  ax[1].plot(x_resnet, mc_dropouts_resnet, label='mc_dropout')
  ax[1].plot(x_resnet, masksembles_resnet, label='masksembles')
  ax[1].set_xlabel("number of dropout/masksembles layers")
  ax[1].set_ylabel(ylabel)
  ax[1].legend()
  ax[1].set_title('ResNet18') 

  ax[2].plot(x_vgg, mc_dropouts_vgg, label='mc_dropout')
  ax[2].plot(x_vgg, masksembles_vgg, label='masksembles')
  ax[2].set_xlabel("number of dropout/masksembles layers")
  ax[2].set_ylabel(ylabel)
  ax[2].legend()
  ax[2].set_title('VGG11')

  plt.show()

if __name__ == '__main__':
  param = sys.argv[1]
  if (param == 'latency'):
    draw_graph(parse_latency_from_xmls, 'latency')
  elif (param == 'LUT'):
    draw_graph(parse_LUT_from_xmls, 'LUT')
  elif (param == 'BRAM'):
    draw_graph(parse_BRAM_18K_from_xmls, 'BRAM')
  elif (param == 'FF'):
    draw_graph(parse_FF_from_xmls, 'FF')
  else:
    print('invalid argument')
