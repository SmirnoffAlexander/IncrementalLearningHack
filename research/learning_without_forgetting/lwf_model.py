import torchvision.transforms as transforms

from run import *
from scrape_images import scraper


class LWFModel:
    def __init__(self, _config):
        self._config = _config
        self.model, _ = get_models(_config)
        self.model = self.model.to(self._config['device'])
        self.transforms = transforms.Compose([transforms.Resize(size=(224, 224)),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.class_names = []  # ["tractor", "lawnmower", "bicycle", "snowboard"]
        self.num_train_imgs = 300

    def add_new_class(self, new_class_name):
        # call scraper and new dir will be created, take data for retraining from this dir
        self.class_names.append(new_class_name)
        #scraper(query=new_class_name, count=self.num_train_imgs)
        self._config['data_path'] = new_class_name

        # update model weights and train model on new class
        # new weights will be saved in preatrined weights
        run_experiment(self._config)

        # update model with new weights from pretrained weights
        self.model, _ = get_models(self._config)
        self.model = self.model.to(self._config['device'])

    def __call__(self, img):
        img = self.transforms(img)
        img = img.unsqueeze(0)
        img = img.to(self._config['device'])
        with torch.no_grad():
            predictions = self.model(img)[0][1000:]
        probs = torch.nn.functional.softmax(predictions)
        print(self.class_names[probs.argmax()], probs)
        return self.class_names[probs.argmax()], probs


if __name__ == "__main__":
    _config = {
        'data_path': 'truck',
        'num_epochs': 20,
        'T': 2,
        'alpha': 0.01,
        'num_new_class': 1,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'lr': 1e-2,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'optimizer': 'SGD',
        'pretrained_weights_path': 'pretrained_weights/net.pth',
        'loss': 'CrossEntropy',
        'batch_size': 4,
        'num_workers': 4,
    }

    from PIL import Image

    class_names = ["tractor", "lawnmower", "bicycle", "snowboard"]
    lwf_model = LWFModel(_config)
    for cl_name in class_names:
        lwf_model.add_new_class(cl_name)

    img = Image.open('tractor/tractor100.jpeg')
    lwf_model(img)
