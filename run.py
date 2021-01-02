import torch
import cv2
import numpy
import shutil
import os
import imageio
import getopt
import PIL
import sys

import softsplat
import flow

from PIL import Image 

if os.path.isdir('summation'):
  shutil.rmtree('summation')
if os.path.isdir('average'):
  shutil.rmtree('average')
if os.path.isdir('linear'):
  shutil.rmtree('linear')
if os.path.isdir('softmax'):
  shutil.rmtree('softmax')
if os.path.isdir('video'):
  shutil.rmtree('video')

os.makedirs('summation')
os.makedirs('average')
os.makedirs('linear')
os.makedirs('softmax')
os.makedirs('video')
##########################################################

assert(int(str('').join(torch.__version__.split('.')[0:2])) >= 13) # requires at least pytorch version 1.3.0

##########################################################

def read_flo(strFile):
  with open(strFile, 'rb') as objFile:
      strFlow = objFile.read()
  # end

  assert(numpy.frombuffer(buffer=strFlow, dtype=numpy.float32, count=1, offset=0) == 202021.25)

  intWidth = numpy.frombuffer(buffer=strFlow, dtype=numpy.int32, count=1, offset=4)[0]
  intHeight = numpy.frombuffer(buffer=strFlow, dtype=numpy.int32, count=1, offset=8)[0]

  return numpy.frombuffer(buffer=strFlow, dtype=numpy.float32, count=intHeight * intWidth * 2, offset=12).reshape([ intHeight, intWidth, 2 ])
# end

##########################################################

backwarp_tenGrid = {}

def backwarp(tenInput, tenFlow):
  if str(tenFlow.shape) not in backwarp_tenGrid:
    tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3]).view(1, 1, 1, -1).expand(-1, -1, tenFlow.shape[2], -1)
    tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).view(1, 1, -1, 1).expand(-1, -1, -1, tenFlow.shape[3])

    backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([ tenHor, tenVer ], 1).cuda()
  # end

  tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)

  return torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=False)
# end

if __name__ == '__main__':

  ##########################################################
  assert(int(str('').join(torch.__version__.split('.')[0:2])) >= 13) # requires at least pytorch version 1.3.0
  torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance
  torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance
  ##########################################################

  arguments_strModel = 'default'
  arguments_strFirst = './images/first.png'
  arguments_strSecond = './images/second.png'
  arguments_strOut = './out.flo'

  for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [ strParameter[2:] + '=' for strParameter in sys.argv[1::2] ])[0]:
    if strOption == '--model' and strArgument != '': arguments_strModel = strArgument # which model to use
    if strOption == '--first' and strArgument != '': arguments_strFirst = strArgument # path to the first frame
    if strOption == '--second' and strArgument != '': arguments_strSecond = strArgument # path to the second frame
    if strOption == '--out' and strArgument != '': arguments_strOut = strArgument # path to where the output should be stored
  # end

  # Opens the Video file
  vidcap = cv2.VideoCapture('chikadance.mp4')
  success,image = vidcap.read()
  i = 0
  while success:
    cv2.imwrite('video/' + f'{i:04d}.jpg', image)    
    success,image = vidcap.read()
    i += 1
  
  vidcap.release()
  cv2.destroyAllWindows()

  pic_1 = cv2.imread(filename='./images/first.png', flags=-1)  
  pic_2 = cv2.imread(filename='./images/second.png', flags=-1)

  tenFirst = torch.FloatTensor(numpy.ascontiguousarray(pic_1[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
  tenSecond = torch.FloatTensor(numpy.ascontiguousarray(pic_2[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
  tenFlow = flow.estimate(tenFirst, tenSecond)

  objOutput = open(arguments_strOut, 'wb')
  numpy.array([80, 73, 69, 72], numpy.uint8).tofile(objOutput)
  numpy.array([tenFlow.shape[2], tenFlow.shape[1]], numpy.int32).tofile(objOutput)
  numpy.array(tenFlow.numpy().transpose(1, 2, 0), numpy.float32).tofile(objOutput)
  objOutput.close()

  tenFirst = torch.FloatTensor(numpy.ascontiguousarray(pic_1.transpose(2, 0, 1)[None, :, :, :].astype(numpy.float32) * (1.0 / 255.0))).cuda()
  tenSecond = torch.FloatTensor(numpy.ascontiguousarray(pic_2.transpose(2, 0, 1)[None, :, :, :].astype(numpy.float32) * (1.0 / 255.0))).cuda()
  tenFlow = torch.FloatTensor(numpy.ascontiguousarray(read_flo('./flow.flo').transpose(2, 0, 1)[None, :, :, :])).cuda()

  tenMetric = torch.nn.functional.l1_loss(input=tenFirst, target=backwarp(tenInput=tenSecond, tenFlow=tenFlow), reduction='none').mean(1, True)

  for intTime, fltTime in enumerate(numpy.linspace(0.0, 1.0, 11).tolist()):
    tenSummation = softsplat.FunctionSoftsplat(tenInput=tenFirst, tenFlow=tenFlow * fltTime, tenMetric=None, strType='summation')
    tenAverage = softsplat.FunctionSoftsplat(tenInput=tenFirst, tenFlow=tenFlow * fltTime, tenMetric=None, strType='average')
    tenLinear = softsplat.FunctionSoftsplat(tenInput=tenFirst, tenFlow=tenFlow * fltTime, tenMetric=(0.3 - tenMetric).clamp(0.0000001, 1.0), strType='linear') # finding a good linearly metric is difficult, and it is not invariant to translations
    tenSoftmax = softsplat.FunctionSoftsplat(tenInput=tenFirst, tenFlow=tenFlow * fltTime, tenMetric=-20.0 * tenMetric, strType='softmax') # -20.0 is a hyperparameter, called 'beta' in the paper, that could be learned using a torch.Parameter
    '''
    img_1 = tenSummation[0, :, :, :].cpu().numpy().transpose(1, 2, 0)
    img_1 = (img_1 * 255).astype(numpy.uint8)
    #cv2_imshow(img_1)
    img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
    img_1 = Image.fromarray(img_1,'RGB')
    fullOutPath = os.path.join('summation', 'summation_'+str(intTime)+'.jpg') 
    img_1.save(fullOutPath, 'JPEG')

    img_2 = tenAverage[0, :, :, :].cpu().numpy().transpose(1, 2, 0)
    img_2 = (img_2 * 255).astype(numpy.uint8)
    #cv2_imshow(img_2)
    img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
    img_2 = Image.fromarray(img_2,'RGB')
    fullOutPath = os.path.join('average', 'average_'+str(intTime)+'.jpg') 
    img_2.save(fullOutPath, 'JPEG')

    img_3 = tenLinear[0, :, :, :].cpu().numpy().transpose(1, 2, 0)
    img_3 = (img_3 * 255).astype(numpy.uint8)
    #cv2_imshow(img_3)
    img_3 = cv2.cvtColor(img_3, cv2.COLOR_BGR2RGB)
    img_3 = Image.fromarray(img_3,'RGB')
    fullOutPath = os.path.join('linear', 'linear_'+str(intTime)+'.jpg') 
    img_3.save(fullOutPath, 'JPEG')
    '''
    img_4 = tenSoftmax[0, :, :, :].cpu().numpy().transpose(1, 2, 0)
    img_4 = (img_4 * 255).astype(numpy.uint8)
    #cv2_imshow(img_4)
    img_4 = cv2.cvtColor(img_4, cv2.COLOR_BGR2RGB)
    img_4 = Image.fromarray(img_4,'RGB')
    fullOutPath = os.path.join('softmax', 'softmax_'+str(intTime)+'.jpg') 
    img_4.save(fullOutPath, 'JPEG')
  # end

  '''
  images_1 = []
  filenames = os.listdir('summation')
  filenames.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
  for filename in filenames:
      images_1.append(imageio.imread('summation/'+filename))
  imageio.mimsave('summation.gif', images_1)

  images_2 = []
  filenames = os.listdir('average')
  filenames.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
  for filename in filenames:
      images_2.append(imageio.imread('average/'+filename))
  imageio.mimsave('average.gif', images_2)

  images_3 = []
  filenames = os.listdir('linear')
  filenames.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
  for filename in filenames:
      images_3.append(imageio.imread('linear/'+filename))
  imageio.mimsave('linear.gif', images_3)
  '''
  images_4 = []
  filenames = os.listdir('softmax')
  filenames.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
  for filename in filenames:
      images_4.append(imageio.imread('softmax/'+filename))
  imageio.mimsave('softmax.gif', images_4)
#end
