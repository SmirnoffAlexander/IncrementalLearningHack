import torchvision.transforms as transforms

from run import *
from scrape_images import scraper


class LWFModel:
    def __init__(self, _config, classes = []):
        self._config = _config
        self.model, _ = get_models(_config)
        self.model = self.model.to(self._config['device'])
        self.transforms = transforms.Compose([transforms.Resize(size=(224, 224)),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.class_names = classes  # ["tractor", "lawnmower", "bicycle", "snowboard"]
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
        self.model.eval()
        img = self.transforms(img)
        img = img.unsqueeze(0)
        img = img.to(self._config['device'])
        with torch.no_grad():
            predictions = self.model(img)[0][1000:]
        print("RN infer finished")
        probs = torch.nn.functional.softmax(predictions).detach().cpu().numpy()
        classes = np.array(self.class_names)
        return classes, probs
