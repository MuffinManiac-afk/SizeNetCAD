from torchvision import models
import torch.nn as nn
import torch
from tqdm import tqdm
import torch.nn.functional as nnf

class clf_model(nn.Module):
    """
    classification model
    """
    def __init__(self,num_classes,learning_rate=0.0001,batch_size=32,model_name='vgg16'):
        """ 
        init cls model

        Args:
            num_classes (_type_): number of classes
            learning_rate (_type_): model learning rate
            batch_size (_type_): model batch size
            model_name (_type_): name of model
        """
        super().__init__()
        self.model_name=model_name
        self.learning_rate=learning_rate
        self.num_classes=num_classes
        if self.model_name=='vgg16':
            self.model= models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            self.model.classifier[6]=nn.Linear(4096,num_classes)
        elif self.model_name=='resnet18':
            self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        elif self.model_name=='resnet152':
            self.model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        elif self.model_name=='convnext':
            self.model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
            self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, num_classes)
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,momentum=0.9)
        self.batch_size=batch_size
        self.learning_rate=learning_rate

    def __call__(self, *args, **kwargs):
        return self.model
    
    def forward(self,image):
        return self.model(image)

    def get_accuracy(self,logit, target, batch_size):
        """
        Obtain accuracy for training round

        Args:
            logit (_type_): predict result
            target (_type_): ground truth
            batch_size (_type_): model batch size

        Returns:
            float: accuracy value
        """
        
        corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
        accuracy = 100.0 * corrects / batch_size
        return accuracy.item()
    def train_model(self,args,dataloader):
        """
        train classification model

        Args:
            dataloader (_type_): pytorch dataloader

        Returns:
           epoch_loss, epoch_ acc
        """
        epoch_acc = 0.0
        device='cuda:0'
        self.model.to(device)
        self.model.train()  # Sets the model in training mode.
        epoch_loss, epoch_correct = 0, 0

        for batch_i, (ids, images, labels) in tqdm(enumerate(dataloader),total=len(dataloader)):
            #labels,images = datas[0], datas[1]
            images, labels = images.to(device), labels.to(device)  # move data to GPU

            # Compute prediction loss
            pred = self.model(images)
            loss = self.criterion(pred, labels)

            # Optimization by gradients
            self.optimizer.zero_grad()  # set prevision gradient to 0
            loss.backward()  # backpropagation to compute gradients
            self.optimizer.step()  # update model params
            epoch_loss += loss.detach().item()
 
            epoch_correct += (torch.max(pred, 1)[1].view(labels.size()).data == labels.data).sum()
            epoch_acc += self.get_accuracy(pred, labels, images.shape[0])

        return epoch_loss / len(dataloader), epoch_acc / len(dataloader)

    def run_validation(self,args, dataloader):
        """
        run validation process

        Args:
            dataloader (_type_): pytorch dataloader

        Returns:
            epoch_loss,epoch_acc
        """
        device = 'cuda:0'
        self.model.to(device)
        self.model.eval()
        epoch_acc = 0.0
        num_batches = len(dataloader)  # batches per epoch
        self.model.to(device)
        epoch_loss, epoch_correct = 0, 0

        with torch.no_grad():
            for batch_i, (ids, images, labels) in tqdm(enumerate(dataloader),total=len(dataloader)):
                #labels,images = datas[0], datas[1]
                images, labels = images.to(device), labels.to(device)
                pred = self.model(images)
                loss = self.criterion(pred, labels)
                epoch_loss += loss.item()
                epoch_acc += self.get_accuracy(pred, labels, images.shape[0])
        return epoch_loss / num_batches, epoch_acc / len(dataloader)

    def get_top5_accuracy(self,logit, target, batch_size,fname):
        """
        Obtain top-5 accuracy for training round

        Args:
            logit (_type_): predict od model
            target (_type_): ground truth
            batch_size (_type_): model batch size
            fname (_type_): file name

        Returns:
            top5_acc,top5_wrong_dict
        """

        corrects = 0
        top5_wrong_dict={}
        prob = nnf.softmax(logit, dim=1)
        prob_value,  first_5_index= prob.topk(5, dim=1)
        for idx, first_5 in enumerate(first_5_index):
            if target.data[idx] in first_5 :
                corrects += 1
            elif target.data[idx] not in first_5 :
                top5_wrong_dict[fname[idx]]=(f'target.data={target.data[idx]},predict_class={first_5},prob_value={prob_value[idx]}')
        top5_accuracy = 100.0 * corrects / batch_size
        return top5_accuracy,top5_wrong_dict
    
    def run_test(self, dataloader,output_path=None):
        """
        run test process

        Args:
            dataloader: pytorch dataloader
            output_path: result path

        Returns:
            top1_acc,top5_acc,top_5_wrong_dict,predict,ground_truth
        """

        self.model.eval()
        epoch_top5_acc = 0.0
        epoch_acc = 0.0
        top5_wrong_dict={}
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_i, (images, labels,fname) in tqdm(enumerate(dataloader),total=len(dataloader)):

                images, labels = images.to(self.device), labels.to(self.device)
                pred = self.model(images)

                pred = torch.nn.functional.softmax(pred, -1)
                one_hot, predict_class = pred.max(dim=-1)
                if self.num_classes >= 5:
                    prob_value, first_5_index = pred.topk(5, dim=1)

                y_pred.extend(predict_class.cpu())  # Save Prediction
                label = labels.data.cpu().numpy()
                y_true.extend(label)  # Save Truth

                epoch_acc += self.get_accuracy(pred, labels, images.shape[0])
                if self.num_classes >= 5:
                    top5_acc ,w = self.get_top5_accuracy(pred, labels, images.shape[0],fname)
                    epoch_top5_acc+=top5_acc
                    top5_wrong_dict=w|top5_wrong_dict
        return epoch_acc / len(dataloader), epoch_top5_acc / len(dataloader),top5_wrong_dict,y_pred,y_true

    def predict(self, dataloader):
        """
        run predict process

        Args:
            dataloader: pytorch dataloader

        Returns:
            ground_truth_list,top_5_index_list,top5_prob_value_list
        """
        self.model.eval()
        y_true_list = []
        prob_value_list=[]
        first_5_index_list=[]
        with torch.no_grad():
            for batch_i, (images, labels,fname,) in tqdm(enumerate(dataloader),total=len(dataloader)):
                images, labels = images.to('cuda'), labels.to('cuda')

                pred = self.model(images)
                pred = torch.nn.functional.softmax(pred, -1)
                one_hot, predict_class = pred.max(dim=-1)
                if self.num_classes >= 5:
                    prob_value, first_5_index = pred.topk(5, dim=1)
                first_5_index_list.extend(first_5_index.cpu())  # Save Prediction
                prob_value_list.extend(prob_value.cpu()) # Save Probability
                label = labels.data.cpu().numpy()
                y_true_list.extend(label)  # Save Truth

        return y_true_list,first_5_index_list,prob_value_list

    def predict_single_image(self, data):
        """
        predict single image

        Args:
            data: image

        Returns:
            first_5_index,top5_prob_value
        """
        self.model.eval()
        with torch.no_grad():
            image = data.to('cuda')
            image = torch.unsqueeze(image, 0)
            pred = self.model(image)
            pred = torch.nn.functional.softmax(pred, -1)
            if self.num_classes >= 5:
                top5_prob_value, first_5_index = pred.topk(5, dim=1)

        return first_5_index, top5_prob_value



