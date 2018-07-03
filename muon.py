from mxnet import gluon,nd,autograd,metric
import matplotlib.pyplot as plt
import numpy as np

class CreateModel(gluon.nn.Block):

    def __init__(self,layer,ctx,precision, **kwargs):
        super(CreateModel, self).__init__(**kwargs)
        
        self.layer=layer
        self.precision=precision
        self.ctx=ctx
        
        self.layer.cast(self.precision)
        
    def grad_check_first_layer(self):
        print( self.layer[0].weight )
        print( self.layer[0].weight.grad().sum() )
        
    def forward(self,x):
        #print(x)
        return self.layer(x)
    
    def fit(self,train_gen,test_gen,epochs,print_every,
            loss_with_softmax,optimizer):
        
        trainer=gluon.Trainer(params=self.collect_params(),
                                   optimizer=optimizer)
        # Initialize some objects for the metrics
        acc=metric.Accuracy()
        train_acc_records=[]
        test_acc_records=[]
        loss_records=[]
        
        for e in range(epochs):
            for i,(data,label) in enumerate(train_gen):

                data=data.as_in_context(self.ctx).astype(self.precision)
                label=label.as_in_context(self.ctx).astype(np.float32)

                with autograd.record():
                    label_linear=self.layer(data)
                    label_linear=label_linear.astype(np.float32) # Improve accuracy, as suggested in nVIDIA's SDK.
                    loss=loss_with_softmax(label_linear,label)
                loss.backward()
                trainer.step(batch_size=128)

                # Print the metrics every several iterations.
                if (i%print_every==0): # print metrics for train (current batch) & test data.
                    label_pred = nd.argmax( nd.softmax(label_linear ), axis=1)
                    acc.reset()
                    acc.update(preds=label_pred, labels=label)
                    train_acc=acc.get()[1]

                    test_acc =self.evaluate_accuracy(test_gen, self.layer)

                    train_acc_records.append(train_acc)
                    test_acc_records.append(test_acc)

                    curr_loss = nd.mean(loss).asscalar()
                    loss_records.append(curr_loss)
                    print("epoch=%2s, iter=%5d, loss=%10f, train acc=%10f, test_acc=%10f"%(e,i,curr_loss,train_acc,test_acc))

        # Visialize the calculated metrics of accuracy during of training.
        self.viz_training(train_acc_records,test_acc_records,loss_records)
        
    def evaluate_accuracy(self,data_iterator, net):
        '''Given model and data, the model accuracy will be calculated.'''
        acc = metric.Accuracy()
        for i, (data, label) in enumerate(data_iterator):
            data = data.as_in_context(self.ctx).astype(self.precision)
            label = label.as_in_context(self.ctx).astype(self.precision)
            output = net(data)
            predictions = nd.argmax(output, axis=1)
            acc.update(preds=predictions, labels=label)
        return acc.get()[1]

    def viz_training(self,train_acc_records,test_acc_records,loss_records):
        """show how the metrics such as loss and model accuracy varies in the progress of training"""

        fig,axes=plt.subplots(1,2,figsize=(18,6),dpi=120)
        axes[0].plot(train_acc_records,ms=5,marker='o',label='train acc',ls='--')
        axes[0].plot(test_acc_records,ms=5,marker='o',label='val acc',ls='--')
        axes[0].legend()
        axes[1].plot(loss_records,ms=5,marker='o',label='train loss',ls='--')
        axes[1].legend()

        for idx,ax in enumerate(axes):
            ax.set_xlabel('Epoch')
            if idx==0:
                ax.set_ylabel('Accuracy')
            else:
                ax.set_ylabel('Loss')
        plt.show()
        
        
class CreateHybridModel(gluon.nn.HybridBlock):

    def __init__(self,layer,ctx,precision, **kwargs):
        super(CreateHybridModel, self).__init__(**kwargs)
        
        self.layer=layer
        self.precision=precision
        self.ctx=ctx
        
        self.layer.cast(self.precision)
        
    def grad_check_first_layer(self):
        print( self.layer[0].weight )
        print( self.layer[0].weight.grad().sum() )
        
    def hybrid_forward(self,F,x):
        #print(x)
        return self.layer(x)
    
    def fit(self,train_gen,test_gen,epochs,print_every,
            loss_with_softmax,optimizer):
        
        trainer=gluon.Trainer(params=self.collect_params(),
                                   optimizer=optimizer)
        # Initialize some objects for the metrics
        acc=metric.Accuracy()
        train_acc_records=[]
        test_acc_records=[]
        loss_records=[]
        
        for e in range(epochs):
            for i,(data,label) in enumerate(train_gen):

                data=data.as_in_context(self.ctx).astype(self.precision)
                label=label.as_in_context(self.ctx).astype(np.float32)

                with autograd.record():
                    label_linear=self.layer(data)
                    label_linear=label_linear.astype(np.float32) # Improve accuracy, as suggested in nVIDIA's SDK.
                    loss=loss_with_softmax(label_linear,label)
                loss.backward()
                trainer.step(batch_size=128)

                # Print the metrics every several iterations.
                if (i%print_every==0): # print metrics for train (current batch) & test data.
                    label_pred = nd.argmax( nd.softmax(label_linear ), axis=1)
                    acc.reset()
                    acc.update(preds=label_pred, labels=label)
                    train_acc=acc.get()[1]

                    test_acc =self.evaluate_accuracy(test_gen, self.layer)

                    train_acc_records.append(train_acc)
                    test_acc_records.append(test_acc)

                    curr_loss = nd.mean(loss).asscalar()
                    loss_records.append(curr_loss)
                    print("epoch=%2s, iter=%5d, loss=%10f, train acc=%10f, test_acc=%10f"%(e,i,curr_loss,train_acc,test_acc))

        # Visialize the calculated metrics of accuracy during of training.
        self.viz_training(train_acc_records,test_acc_records,loss_records)
        
    def evaluate_accuracy(self,data_iterator, net):
        '''Given model and data, the model accuracy will be calculated.'''
        acc = metric.Accuracy()
        for i, (data, label) in enumerate(data_iterator):
            data = data.as_in_context(self.ctx).astype(self.precision)
            label = label.as_in_context(self.ctx).astype(self.precision)
            output = net(data)
            predictions = nd.argmax(output, axis=1)
            acc.update(preds=predictions, labels=label)
        return acc.get()[1]

    def viz_training(self,train_acc_records,test_acc_records,loss_records):
        """show how the metrics such as loss and model accuracy varies in the progress of training"""

        fig,axes=plt.subplots(1,2,figsize=(18,6),dpi=120)
        axes[0].plot(train_acc_records,ms=5,marker='o',label='train acc',ls='--')
        axes[0].plot(test_acc_records,ms=5,marker='o',label='val acc',ls='--')
        axes[0].legend()
        axes[1].plot(loss_records,ms=5,marker='o',label='train loss',ls='--')
        axes[1].legend()

        for idx,ax in enumerate(axes):
            ax.set_xlabel('Epoch')
            if idx==0:
                ax.set_ylabel('Accuracy')
            else:
                ax.set_ylabel('Loss')
        plt.show()