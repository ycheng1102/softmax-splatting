import torch
import cv2
import numpy
import shutil
import os
import imageio
import getopt
import PIL
import sys
import math

import softsplat
import flow

from PIL import Image 
from google.colab.patches import cv2_imshow

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

  arg_width = 640
  arg_height = 360
  arg_FPS = 'default'
  arg_Flow = './out.flo'

  for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [ strParameter[2:] + '=' for strParameter in sys.argv[1::2] ])[0]:
    if strOption == '--video' and strArgument != '': arg_video = strArgument
    if strOption == '--width' and strArgument != '': arg_width = strArgument
    if strOption == '--height' and strArgument != '': arg_height = strArgument
    if strOption == '--fps' and strArgument != '': arg_FPS = strArgument
    if strOption == '--flow' and strArgument != '': arg_Flow = strArgument
  # end

  arg_FPS = int(arg_FPS)

  time = numpy.linspace(0.0, 1.0, arg_FPS + 1).tolist()
  time.remove(0.0)
  time.remove(1.0)
  #print(time)

  # Opens the Video file
  vidcap = cv2.VideoCapture(arg_video)
  FPS = math.ceil(vidcap.get(cv2.CAP_PROP_FPS))

  assert FPS < arg_FPS * FPS, "Original FPS is bigger than the desired FPS!"
  print('Convert from %dfps to %dfps.' % (FPS, arg_FPS * FPS))
  print('Capture frame from video...')

  success,img = vidcap.read()
  img = cv2.resize(img,(arg_width, arg_height),interpolation=cv2.INTER_LANCZOS4)
  i = 0
  while success:
    img = cv2.resize(img,(arg_width, arg_height),interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite('video/' + f'{i:04d}.png', img)    
    success,img = vidcap.read()
    i += 1
  # end
  vidcap.release()
  cv2.destroyAllWindows()
  
  img_array = []
  listA = os.listdir('video')
  listB = os.listdir('video')
  listA.sort()
  listB.sort()
  listB.remove('0000.png')
  filecounts = len(listA)

  print()
  i = 0
  for A, B in zip(listA, listB):

    #if i == 30:
    #  break
    if i % (arg_FPS * FPS) == 0: print('-- processing %d / %d' % (i, filecounts))
    i += 1

    fullpath_A = 'video/' + A
    fullpath_B = 'video/' + B
    pic_A = cv2.imread(filename=fullpath_A, flags=-1)
    pic_B = cv2.imread(filename=fullpath_B, flags=-1)

    tenFirst = torch.FloatTensor(numpy.ascontiguousarray(pic_A[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
    tenSecond = torch.FloatTensor(numpy.ascontiguousarray(pic_B[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
    tenFlow = flow.estimate(tenFirst, tenSecond)

    objOutput = open(arg_Flow, 'wb')
    numpy.array([80, 73, 69, 72], numpy.uint8).tofile(objOutput)
    numpy.array([tenFlow.shape[2], tenFlow.shape[1]], numpy.int32).tofile(objOutput)
    numpy.array(tenFlow.numpy().transpose(1, 2, 0), numpy.float32).tofile(objOutput)
    objOutput.close()

    tenFirst = torch.FloatTensor(numpy.ascontiguousarray(pic_A.transpose(2, 0, 1)[None, :, :, :].astype(numpy.float32) * (1.0 / 255.0))).cuda()
    tenSecond = torch.FloatTensor(numpy.ascontiguousarray(pic_B.transpose(2, 0, 1)[None, :, :, :].astype(numpy.float32) * (1.0 / 255.0))).cuda()
    tenFlow = torch.FloatTensor(numpy.ascontiguousarray(read_flo(arg_Flow).transpose(2, 0, 1)[None, :, :, :])).cuda()

    tenMetric = torch.nn.functional.l1_loss(input=tenFirst, target=backwarp(tenInput=tenSecond, tenFlow=tenFlow), reduction='none').mean(1, True)

    img_array.append(pic_A)
    for t in time:
      tenSoftmax = softsplat.FunctionSoftsplat(tenInput=tenFirst, tenFlow=tenFlow * t, tenMetric=-20.0 * tenMetric, strType='softmax')
      img = tenSoftmax[0, :, :, :].cpu().numpy().transpose(1, 2, 0)
      img = (img * 255).astype(numpy.uint8)
      img_array.append(img)
  # end

  print('-- processing %d / %d' % (filecounts, filecounts))
  print('Creating video...')

  out = cv2.VideoWriter('project.mp4',cv2.VideoWriter_fourcc(*'mp4v'), arg_FPS * FPS , (arg_width, arg_height))

  print(len(img_array))
  for i in range(len(img_array)):
    out.write(img_array[i])
  out.release()

  print('Done!')