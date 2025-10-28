import numpy as np
import cv2 as cv
import time
import math
from PIL import Image
from tinygrad.tensor import Tensor
from tinygrad.helpers import fetch
from tinygrad.nn import BatchNorm2d, Conv2d

def parse_cfg(cfg):
  # Return a list of blocks
  lines = cfg.decode("utf-8").split('\n')
  lines = [x for x in lines if len(x) > 0]
  lines = [x for x in lines if x[0] != '#']
  lines = [x.rstrip().lstrip() for x in lines]
  block, blocks = {}, []
  for line in lines:
    if line[0] == "[":
      if len(block) != 0:
        blocks.append(block)
        block = {}
      block["type"] = line[1:-1].rstrip()
    else:
      key,value = line.split("=")
      block[key.rstrip()] = value.lstrip()
  blocks.append(block)
  return blocks

def predict_transform(prediction, inp_dim, anchors, num_classes):
  batch_size = prediction.shape[0]
  stride = inp_dim // prediction.shape[2]
  grid_size = inp_dim // stride
  bbox_attrs = 5 + num_classes
  num_anchors = len(anchors)
  prediction = prediction.reshape(shape=(batch_size, bbox_attrs*num_anchors, grid_size*grid_size))
  prediction = prediction.transpose(1, 2)
  prediction = prediction.reshape(shape=(batch_size, grid_size*grid_size*num_anchors, bbox_attrs))
  prediction_cpu = prediction.numpy()
  for i in (0, 1, 4):
    prediction_cpu[:,:,i] = 1 / (1 + np.exp(-prediction_cpu[:,:,i]))
  # Add the center offsets
  grid = np.arange(grid_size)
  a, b = np.meshgrid(grid, grid)
  x_offset = a.reshape((-1, 1))
  y_offset = b.reshape((-1, 1))
  x_y_offset = np.concatenate((x_offset, y_offset), 1)
  x_y_offset = np.tile(x_y_offset, (1, num_anchors))
  x_y_offset = x_y_offset.reshape((-1,2))
  x_y_offset = np.expand_dims(x_y_offset, 0)
  anchors = [(a[0]/stride, a[1]/stride) for a in anchors]
  anchors = np.tile(anchors, (grid_size*grid_size, 1))
  anchors = np.expand_dims(anchors, 0)
  prediction_cpu[:,:,:2] += x_y_offset
  prediction_cpu[:,:,2:4] = np.exp(prediction_cpu[:,:,2:4])*anchors
  prediction_cpu[:,:,5:5+num_classes] = 1 / (1 + np.exp(-prediction_cpu[:,:,5:5+num_classes]))
  prediction_cpu[:,:,:4] *= stride
  return Tensor(prediction_cpu)

def process_results(prediction, confidence=0.9, num_classes=80, nms_conf=0.4):
  prediction = prediction.detach().numpy()
  conf_mask = (prediction[:,:,4] > confidence)
  conf_mask = np.expand_dims(conf_mask, 2)
  prediction = prediction * conf_mask
  # Non max suppression
  box_corner = prediction
  box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
  box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
  box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2)
  box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
  prediction[:,:,:4] = box_corner[:,:,:4]
  write = False
  # Process img
  img_pred = prediction[0]
  max_conf = np.amax(img_pred[:,5:5+num_classes], axis=1)
  max_conf_score = np.argmax(img_pred[:,5:5+num_classes], axis=1)
  max_conf_score = np.expand_dims(max_conf_score, axis=1)
  max_conf = np.expand_dims(max_conf, axis=1)
  seq = (img_pred[:,:5], max_conf, max_conf_score)
  image_pred = np.concatenate(seq, axis=1)
  non_zero_ind = np.nonzero(image_pred[:,4])[0]
  assert all(image_pred[non_zero_ind,0] > 0)
  image_pred_ = np.reshape(image_pred[np.squeeze(non_zero_ind),:], (-1, 7))
  if image_pred_.shape[0] == 0:
    print("No detections found!")
    return 0
  for cls in np.unique(image_pred_[:, -1]):
    # perform NMS, get the detections with one particular class
    cls_mask = image_pred_*np.expand_dims(image_pred_[:, -1] == cls, axis=1)
    class_mask_ind = np.squeeze(np.nonzero(cls_mask[:,-2]))
    # class_mask_ind = np.nonzero()
    image_pred_class = np.reshape(image_pred_[class_mask_ind], (-1, 7))
    # sort the detections such that the entry with the maximum objectness
    # confidence is at the top
    conf_sort_index = np.argsort(image_pred_class[:,4])
    image_pred_class = image_pred_class[conf_sort_index]
    for i in range(image_pred_class.shape[0]):
      # Get the IOUs of all boxes that come after the one we are looking at in the loop
      try:
        ious = bbox_iou(np.expand_dims(image_pred_class[i], axis=0), image_pred_class[i+1:])
      except:
        break
      # Zero out all the detections that have IoU > threshold
      iou_mask = np.expand_dims((ious < nms_conf), axis=1)
      image_pred_class[i+1:] *= iou_mask
      # Remove the non-zero entries
      non_zero_ind = np.squeeze(np.nonzero(image_pred_class[:,4]))
      image_pred_class = np.reshape(image_pred_class[non_zero_ind], (-1, 7))
    batch_ind = np.array([[0]])
    seq = (batch_ind, image_pred_class)
    if not write:
      output, write = np.concatenate(seq, axis=1), True
    else:
      out = np.concatenate(seq, axis=1)
      output = np.concatenate((output,out))
  return output

def bbox_iou(box1, box2):
  """
  Returns the IoU of two bounding boxes
  IoU: IoU = Area Of Overlap / Area of Union -> How close the predicted bounding box is
  to the ground truth bounding box. Higher IoU = Better accuracy
  In training, used to track accuracy. with inference, using to remove duplicate bounding boxes
  """
  # Get the coordinates of bounding boxes
  b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
  b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
  # get the coordinates of the intersection rectangle
  inter_rect_x1 = np.maximum(b1_x1, b2_x1)
  inter_rect_y1 = np.maximum(b1_y1, b2_y1)
  inter_rect_x2 = np.minimum(b1_x2, b2_x2)
  inter_rect_y2 = np.minimum(b1_y2, b2_y2)
  #Intersection area
  inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, 99999) * np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0, 99999)
  #Union Area
  b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
  b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)
  iou = inter_area / (b1_area + b2_area - inter_area)
  return iou

def add_boxes(img, prediction):
  if isinstance(prediction, int): # no predictions
    return img
  coco_labels = fetch('https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names').read_bytes()
  coco_labels = coco_labels.decode('utf-8').split('\n')
  height, width = img.shape[0:2]
  scale_factor = 608 / width
  prediction[:,[1,3]] -= (608 - scale_factor * width) / 2
  prediction[:,[2,4]] -= (608 - scale_factor * height) / 2
  for pred in prediction:
    corner1 = tuple(pred[1:3].astype(int))
    corner2 = tuple(pred[3:5].astype(int))
    w = corner2[0] - corner1[0]
    h = corner2[1] - corner1[1]
    corner2 = (corner2[0] + w, corner2[1] + h)
    label = coco_labels[int(pred[-1])]
    img = cv.rectangle(img, corner1, corner2, (255, 0, 0), 2)
    t_size = cv.getTextSize(label, cv.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = corner1[0] + t_size[0] + 3, corner1[1] + t_size[1] + 4
    img = cv.rectangle(img, corner1, c2, (255, 0, 0), -1)
    img = cv.putText(img, label, (corner1[0], corner1[1] + t_size[1] + 4), cv.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
  return img

class Darknet:
  def __init__(self, cfg):
    self.blocks = parse_cfg(cfg)
    self.net_info, self.module_list = self.create_modules(self.blocks)
    print("Modules length:", len(self.module_list))

  def create_modules(self, blocks):
    net_info = blocks[0] # Info about model hyperparameters
    prev_filters, filters = 3, None
    output_filters, module_list = [], []
    ## module
    for index, x in enumerate(blocks[1:]):
      module_type = x["type"]
      module = []
      if module_type == "convolutional":
        try:
          batch_normalize, bias = int(x["batch_normalize"]), False
        except:
          batch_normalize, bias = 0, True
        # layer
        activation = x["activation"]
        filters = int(x["filters"])
        padding = int(x["pad"])
        pad = (int(x["size"]) - 1) // 2 if padding else 0
        module.append(Conv2d(prev_filters, filters, int(x["size"]), int(x["stride"]), pad, bias=bias))
        # BatchNorm2d
        if batch_normalize:
          module.append(BatchNorm2d(filters, eps=1e-05, track_running_stats=True))
        # LeakyReLU activation
        if activation == "leaky":
          module.append(lambda x: x.leaky_relu(0.1))
      elif module_type == "maxpool":
        size, stride = int(x["size"]), int(x["stride"])
        module.append(lambda x: x.max_pool2d(kernel_size=(size, size), stride=stride))
      elif module_type == "upsample":
        module.append(lambda x: Tensor(x.numpy().repeat(2, axis=-2).repeat(2, axis=-1)))
      elif module_type == "route":
        x["layers"] = x["layers"].split(",")
        # Start of route
        start = int(x["layers"][0])
        # End if it exists
        try:
          end = int(x["layers"][1])
        except:
          end = 0
        if start > 0: start -= index
        if end > 0: end -= index
        module.append(lambda x: x)
        if end < 0:
          filters = output_filters[index + start] + output_filters[index + end]
        else:
          filters = output_filters[index + start]
      # Shortcut corresponds to skip connection
      elif module_type == "shortcut":
        module.append(lambda x: x)
      elif module_type == "yolo":
        mask = list(map(int, x["mask"].split(",")))
        anchors = [int(a) for a in x["anchors"].split(",")]
        anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
        module.append([anchors[i] for i in mask])
      # Append to module_list
      module_list.append(module)
      if filters is not None:
        prev_filters = filters
      output_filters.append(filters)
    return (net_info, module_list)

  def dump_weights(self):
    for i in range(len(self.module_list)):
      module_type = self.blocks[i + 1]["type"]
      if module_type == "convolutional":
        print(self.blocks[i + 1]["type"], "weights", i)
        model = self.module_list[i]
        conv = model[0]
        print(conv.weight.numpy()[0][0][0])
        if conv.bias is not None:
          print("biases")
          print(conv.bias.shape)
          print(conv.bias.numpy()[0][0:5])
        else:
          print("None biases for layer", i)

  def load_weights(self, url):
    weights = np.frombuffer(fetch(url).read_bytes(), dtype=np.float32)[5:]
    ptr = 0
    for i in range(len(self.module_list)):
      module_type = self.blocks[i + 1]["type"]
      if module_type == "convolutional":
        model = self.module_list[i]
        try: # we have batchnorm, load conv weights without biases, and batchnorm values
          batch_normalize = int(self.blocks[i+1]["batch_normalize"])
        except: # no batchnorm, load conv weights + biases
          batch_normalize = 0
        conv = model[0]
        if batch_normalize:
          bn = model[1]
          # Get the number of weights of batchnorm
          num_bn_biases = math.prod(bn.bias.shape)
          # Load weights
          bn_biases = Tensor(weights[ptr:ptr + num_bn_biases].astype(np.float32))
          ptr += num_bn_biases
          bn_weights = Tensor(weights[ptr:ptr+num_bn_biases].astype(np.float32))
          ptr += num_bn_biases
          bn_running_mean = Tensor(weights[ptr:ptr+num_bn_biases].astype(np.float32))
          ptr += num_bn_biases
          bn_running_var = Tensor(weights[ptr:ptr+num_bn_biases].astype(np.float32))
          ptr += num_bn_biases
          # Cast the loaded weights into dims of model weights
          bn_biases = bn_biases.reshape(shape=tuple(bn.bias.shape))
          bn_weights = bn_weights.reshape(shape=tuple(bn.weight.shape))
          bn_running_mean = bn_running_mean.reshape(shape=tuple(bn.running_mean.shape))
          bn_running_var = bn_running_var.reshape(shape=tuple(bn.running_var.shape))
          # Copy data
          bn.bias = bn_biases
          bn.weight = bn_weights
          bn.running_mean = bn_running_mean
          bn.running_var = bn_running_var
        else:
          # load biases of the conv layer
          num_biases = math.prod(conv.bias.shape)
          # Load weights
          conv_biases = Tensor(weights[ptr: ptr+num_biases].astype(np.float32))
          ptr += num_biases
          # Reshape
          conv_biases = conv_biases.reshape(shape=tuple(conv.bias.shape))
          # Copy
          conv.bias = conv_biases
        # Load weighys for conv layers
        num_weights = math.prod(conv.weight.shape)
        conv_weights = Tensor(weights[ptr:ptr+num_weights].astype(np.float32))
        ptr += num_weights
        conv_weights = conv_weights.reshape(shape=tuple(conv.weight.shape))
        conv.weight = conv_weights

  def forward(self, x):
    modules = self.blocks[1:]
    outputs = {} # Cached outputs for route layer
    detections, write = None, False
    for i, module in enumerate(modules):
      module_type = (module["type"])
      if module_type == "convolutional" or module_type == "upsample":
        for layer in self.module_list[i]:
          x = layer(x)
      elif module_type == "route":
        layers = module["layers"]
        layers = [int(a) for a in layers]
        if (layers[0]) > 0:
          layers[0] = layers[0] - i
        if len(layers) == 1:
          x = outputs[i + (layers[0])]
        else:
          if (layers[1]) > 0: layers[1] = layers[1] - i
          map1 = outputs[i + layers[0]]
          map2 = outputs[i + layers[1]]
          x = Tensor(np.concatenate((map1.numpy(), map2.numpy()), axis=1))
      elif module_type == "shortcut":
        from_ = int(module["from"])
        x = outputs[i - 1] + outputs[i + from_]
      elif module_type == "yolo":
        anchors = self.module_list[i][0]
        inp_dim = int(self.net_info["height"])  # 416
        num_classes = int(module["classes"])
        x = predict_transform(x, inp_dim, anchors, num_classes)
        if not write:
          detections, write = x, True
        else:
          detections = Tensor(np.concatenate((detections.numpy(), x.numpy()), axis=1))
      outputs[i] = x
    return detections

def letterbox_cv2(img, new_shape=(608, 608), color=(114, 114, 114)):
  h0, w0 = img.shape[:2]
  nh, nw = new_shape
  r = min(nw / w0, nh / h0)
  new_w, new_h = int(round(w0*r)), int(round(h0*r))
  interp = cv.INTER_AREA if r < 1 else cv.INTER_LINEAR
  img_resized = cv.resize(img, (new_w, new_h), interpolation=interp)
  
  dw, dh = nw-new_w, nh-new_h
  left = int(round(dw / 2.0))
  top = int(round(dh / 2.0))
  right = dw - left
  bottom = dh - top
  img_padded = cv.copyMakeBorder(img_resized, top, bottom, left, right,
                                 borderType=cv.BORDER_CONSTANT, value=color)
  return img_padded, r, (left, top)



def prepare_for_yolo(img):
  img_padded, scale, (pad_x, pad_y) = letterbox_cv2(img)
  img_padded = img_padded[:, :, ::-1] # bgr -> rgb
  img_float = img_padded.astype(np.float32) / 255.0 # -> [0, 1]
  img_input = np.ascontiguousarray(img_float.transpose(2, 0, 1)[None, ...]) # h w c -> c h w -> 1 c h w
  print(img.shape, img_padded.shape, img_input.shape)
  return img_input, scale, (pad_x, pad_y)
  
  
  
def infer(model, img):
  '''
  model need size (608, 608)
  img size (480, 640, 3)
  '''
  frame, scale, pad = prepare_for_yolo(img)
  pred = model.forward(Tensor(frame.astype(np.float32)))
  return pred
  
  



if __name__ == "__main__":
  model = Darknet(fetch('https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg').read_bytes())
  print("Loading weights file (237MB). This might take a while…")
  model.load_weights('https://github.com/shadiakiki1986/yolov3.weights/releases/download/3.0.1/yolov3.weights')


  cap = cv.VideoCapture(0)
  cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
  cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

  if not cap.isOpened():
    print("cannot open camera")
    exit()

  while True:
    ret, frame = cap.read()
    if not ret:
      print("can't receive frame")
      break

    # prepare_for_yolo(frame) # work
    
    infer(model, frame)
    prediction = process_results(infer(model, frame))
    img = Image.fromarray(frame[:, :, [2,1,0]])
    boxes = add_boxes(np.array(img.resize((608, 608))), prediction)
    boxes = cv.cvtColor(boxes, cv.COLOR_RGB2BGR)
    cv.imshow('yolo', boxes)
    
    gray =  cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # cv.imshow("frame", gray)
    if cv.waitKey(1) == ord('q'):
      break
    
  cap.release()
  cv.destroyAllWindows()