import matplotlib.pyplot as plt
import numpy as np
from mxnet.gluon import nn,Block
from mxnet import nd,gpu
from mxnet.contrib.ndarray import MultiBoxPrior
import pandas as pd

def class_predictor(num_anchors, num_classes):
    """return a layer to predict classes"""
    return nn.Conv2D(num_anchors * (num_classes + 1), 3, padding=1)

def box_predictor(num_anchors):
    """return a layer to predict delta locations"""
    return nn.Conv2D(num_anchors * 4, 3, padding=1)

def down_sample(num_filters):
    """stack two Conv-BatchNorm-Relu blocks and then a pooling layer 
    to halve the feature size"""
    out = nn.HybridSequential()
    for _ in range(2):
        out.add(nn.Conv2D(num_filters, 3, strides=1, padding=1))
        out.add(nn.BatchNorm(in_channels=num_filters))
        out.add(nn.Activation('relu'))
    out.add(nn.MaxPool2D(2))
    return out

def body():
    """return the body network"""
    out = nn.HybridSequential()
    for nfilters in [16, 32, 64]:
        out.add(down_sample(nfilters))
    return out

def intersect(box_a, box_b):
    """
    We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """

    A = box_a.shape[0]
    B = box_b.shape[0]
    max_xy = nd.minimum(box_a[:, 2:].expand_dims(axis=1)
                                    .repeat(axis=1,repeats=B), 
                        box_b[:, 2:].expand_dims(axis=0)
                                    .repeat(axis=0,repeats=A) 
                       )
    min_xy = nd.maximum(box_a[:, :2].expand_dims(axis=1)
                                    .repeat(axis=1,repeats=B),
                        box_b[:, :2].expand_dims(axis=0)
                                    .repeat(axis=0,repeats=A)
                       )
    inter = nd.clip((max_xy - min_xy),0,np.nan)
    return inter[:, :, 0] * inter[:, :, 1]

def iou(box_a, box_b):
    """
       Compute the Intersection over Union ( or the so called jaccard overlap) of two sets of boxes.  
       Here we operate on ground truth boxes and default boxes.
       E.g.:
          A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
       Args:
          box_a: (tensor) Ground truth bounding boxes, 
                 Shape:    [num_objects,4]
          box_b: (tensor) Prior boxes from priorbox layers, 
                 Shape: [num_priors,4]
       Return:
          jaccard overlap: (tensor) 
                           Shape: [box_a.size(0), box_b.size(0)]
    """
    A = box_a.shape[0]
    B = box_b.shape[0]
    
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1])    ).expand_dims(axis=1) \
                                              .repeat(axis=1,repeats=B)
    area_b = ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1])    ).expand_dims(axis=0) \
                                              .repeat(axis=0,repeats=A)
    union = area_a + area_b - inter
    return inter / union

def loc_difference_calculator(anchor_locs,label_locs,variances=(0.1, 0.1, 0.4, 0.4)):
    '''計算真實方框和預設錨框的位置差距。'''

    al = anchor_locs[:,0] # anchor box; left (or: x_min)
    at = anchor_locs[:,1] # anchor box; top (or: y_min)
    ar = anchor_locs[:,2] # anchor box; right (or: x_max)
    ab = anchor_locs[:,3] # anchor box; bottom (or: y_max)

    gl = label_locs[:,0] # ground truth box; left (or: x_min)
    gt = label_locs[:,1] # ground truth box; top (or: y_min)
    gr = label_locs[:,2] # ground truth box; right (or: x_max)
    gb = label_locs[:,3] # ground truth box; bottom (or: y_max)

    # 對於預設錨框，將x_min,y_min,x_max,y_max 轉換成 cx(中心點的x),cy(中心點的y),w(矩形的寬),h(矩形的高)
    aw = ar - al
    ah = ab - at
    ax = 0.5 * ( al + ar )
    ay = 0.5 * ( at + ab )
    
    # 對於真實方框，將x_min,y_min,x_max,y_max 轉換成 cx(中心點的x),cy(中心點的y),w(矩形的寬),h(矩形的高)
    gw = gr - gl
    gh = gb - gt
    gx = 0.5 * ( gl + gr )
    gy = 0.5 * ( gt + gb )

    # 計算預設錨框與真實方框的位置差距。 於矩形的寬和高方面，我們希望機器不要太注重。
    # 故，我們以取log的方式，來縮減矩形寬和高的差距。
    dx = (gx - ax) / aw    / variances[0]
    dy = (gy - ay) / ah    / variances[1]
    dw = nd.log( gw / aw ) / variances[2]
    dh = nd.log( gh / ah ) / variances[3]
    return nd.stack(dx,dy,dw,dh,axis=1) 

def MultiBoxTarget(anchors, class_predictions, labels, hard_neg_ratio=3, ctx=gpu(),verbose=False):
    '''將真實方框(ground truth boxes)和預設錨框(default anchor boxes)做配對。
       labels:         真實方框 (ground truth boxes)。
       anchors:        預設錨框 (default anchor boxes)。
       hard_neg_ratio: 負樣本(背景)和正樣本(有物體的錨框數)的比例。預設是3:1。
    '''

    if verbose:
        print("anchors shape=", anchors.shape)
    assert anchors.shape[0]==1

    batch_size=len(labels)
    num_priors=anchors.shape[1]

    if verbose:
        print("batch size=\t",batch_size)
        print("num priors=\t",num_priors)

    anchor_shifts = nd.zeros( (batch_size, anchors.shape[1] * 4) ,ctx=ctx)
    box_mask = nd.zeros( (batch_size, anchors.shape[1] * 4)  ,ctx=ctx)

    anchor_classes = nd.zeros( (batch_size, num_priors)  ,ctx=ctx) -1
    anchor_indices = nd.zeros( (batch_size, num_priors)  ,ctx=ctx) -1

    classes_mask = nd.zeros( (batch_size, anchors.shape[1])  ,ctx=ctx)
    
    shifts_tmp = nd.zeros_like( anchors[0] )
    mask_tmp = nd.zeros_like( anchors[0] )
    
    for i in range(batch_size):
       
        label_locs = labels[i][:,1:]
        label_classes = labels[i][:,0]
        class_preds = class_predictions[i]
        # obtain IoU
        ious = iou( label_locs , anchors[0] )
        # identify how many ground truth objects are there in this batch
        if -1 in label_classes:
            num_obj = label_classes.argmin(axis=0).astype("int32").asscalar()
        else:
            num_obj = label_classes.size
        if num_obj == 0:
            continue
        # matching anchor boxes with ground truth boxes
        ious_flag = ious > 0.5
        # find locations of the best priors
        best_prior_idx = ious[0:num_obj,:].argmax(axis=1)

        idx_row=[*range(best_prior_idx.size)]
        ious_flag[idx_row,best_prior_idx] += 1

        if_negative = label_classes != -1
        label_classes = label_classes + if_negative * 1
        # add the -1 class to the end
        label_classes=nd.concat(label_classes, nd.array([-1],ctx=label_classes.context),dim=0)
        if verbose:
            print("label classes=\t",label_classes)
        if_matched = ious_flag.sum(axis=0) != 0
        label_classes_last_idx= len(label_classes) - 1
        anchor_indices[i] = ( ious_flag.argmax(axis=0) - label_classes_last_idx ) * if_matched + label_classes_last_idx
        anchor_classes[i] = label_classes[anchor_indices[i]]
        
        # find indices of the negative anchors
        neg_indices, = np.where( anchor_classes[i].asnumpy() == -1)
        # count number of positive/negative/hard negative anchors
        num_neg_anchors = len(neg_indices)
        num_pos_anchors = num_priors - num_neg_anchors
        num_hard_neg_anchors = num_pos_anchors * hard_neg_ratio
        
        # hard negative mining
        #conf_loss_indices = nd.argsort( (-nd.softmax(  class_preds[neg_indices] ) )[:,0], is_ascend=False ).astype("int32")
        conf_loss_indices = nd.argsort( (-  class_preds[neg_indices] )[:,0], is_ascend=False ).astype("int32")
        neg_indices_sorted = nd.array(neg_indices,ctx=class_preds.context,dtype="int32")[conf_loss_indices]
        hard_neg_indices=neg_indices_sorted[:num_hard_neg_anchors]

        anchor_classes[i][hard_neg_indices] = 0

        # find indices of the positive anchors
        pos_indices,= np.where( anchor_indices[i].asnumpy() < label_classes_last_idx )
        
        pos_indices = nd.array(pos_indices,ctx=hard_neg_indices.context).astype("int32")

        cls_indices = nd.concatenate([hard_neg_indices,pos_indices]) # store indices for classification.
        classes_mask[i][cls_indices] = 1
        
        if verbose:
            print("========================================")
            display(pd.DataFrame(anchor_classes[i].asnumpy()).groupby(0).indices)
            df=pd.DataFrame(anchor_classes[i].asnumpy()).reset_index().groupby(0).count()
            df.index.name = "class"
            df.index=df.index.astype("int32")
            df.columns=["number of default anchor boxes"]
            display(df)
            print("========================================")

        # obtain locations of the positve anchors
        pos_anchors_loc=anchors[0][pos_indices]
        # obtain indices of the ground truth labels
        idx_gt=anchor_indices[i][pos_indices]
        # obtain locations of the ground truth labels
        labels_loc=label_locs[idx_gt]
        assert len(pos_anchors_loc) == len(labels_loc)
        # calculate location differences between ground truth labels and positive anchors
        shifts_tmp[pos_indices]=loc_difference_calculator(pos_anchors_loc,labels_loc)
        mask_tmp[pos_indices]=1.
        
        anchor_shifts[i]=shifts_tmp.reshape(-1)
        box_mask[i]=mask_tmp.reshape(-1)
                
    return anchor_shifts,box_mask,anchor_classes,classes_mask
