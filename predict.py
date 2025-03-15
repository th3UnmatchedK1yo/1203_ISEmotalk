import os 
import numpy
import extract_feature as ef
import model
import utils

def predict (config, audio_path: str,cls )->None:
    data_path=ef.get_path(audio_path,train=False)
    test_feature=ef.get_features(data_path,config,train=False)

    result= model.predict(test_feature)
    result_prob= model.predict_proba(test_feature)

    print('Recogntion: ', config.dataset.class_labels[int(result)])
    print('Probability: ', result_prob)
    utils.radar(result_prob,config.dataset.class_labels)

if __name__ == '__main__':
    audio_path = ''
    config = utils.parse_opt()
    model = model.load(config)
    predict(config, audio_path, model)