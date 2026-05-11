import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,f1_score,precision_score,recall_score,
    confusion_matrix,ConfusionMatrixDisplay,roc_curve,auc
)

# ================= CONFIG =================

DATASET_PATH="/content/dataset"
IMG_SIZE=(128,128)
LATENT_DIM=64
BATCH_SIZE=32
EPOCHS=80
BETA=4.0

USE_PERCEPTUAL_LOSS=True
USE_FFT_LOSS=True

# ================= DATASET =================

def load_dataset(dataset_path):

    images,labels=[],[]

    class_names=sorted([
        folder for folder in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path,folder))
    ])

    print("\nDetected Classes:")
    print(class_names)

    total_images=0

    for label,class_name in enumerate(class_names):

        class_path=os.path.join(dataset_path,class_name)

        image_files=os.listdir(class_path)

        print(f"{class_name}: {len(image_files)} images")

        total_images+=len(image_files)

        for image_name in image_files:

            image_path=os.path.join(class_path,image_name)

            try:
                image=load_img(image_path,target_size=IMG_SIZE)

                image=img_to_array(image)/255.0

                images.append(image)

                labels.append(label)

            except:
                pass

    images=np.array(images,dtype=np.float32)
    labels=np.array(labels)

    print(f"\nTotal Images: {len(images)}")

    return images,labels,class_names

# ================= SAMPLING =================

class Sampling(tf.keras.layers.Layer):

    def call(self,inputs):

        z_mean,z_log_var=inputs

        batch=tf.shape(z_mean)[0]
        dim=tf.shape(z_mean)[1]

        epsilon=tf.random.normal(shape=(batch,dim))

        return z_mean+tf.exp(0.5*z_log_var)*epsilon

# ================= FEATURE EXTRACTOR =================

def build_feature_extractor():

    vgg=tf.keras.applications.VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=(128,128,3)
    )

    vgg.trainable=False

    output=vgg.get_layer("block3_conv3").output

    return Model(vgg.input,output)

feature_extractor=build_feature_extractor()

# ================= LOSSES =================

def reconstruction_loss(y_true,y_pred):

    return tf.reduce_mean(
        tf.reduce_sum(
            tf.square(y_true-y_pred),
            axis=(1,2,3)
        )
    )

def kl_divergence_loss(z_mean,z_log_var):

    return -0.5*tf.reduce_mean(
        tf.reduce_sum(
            1+z_log_var-tf.square(z_mean)-tf.exp(z_log_var),
            axis=1
        )
    )

def perceptual_loss(y_true,y_pred):

    y_true=tf.keras.applications.vgg16.preprocess_input(y_true*255.0)
    y_pred=tf.keras.applications.vgg16.preprocess_input(y_pred*255.0)

    true_features=feature_extractor(y_true)
    pred_features=feature_extractor(y_pred)

    return tf.reduce_mean(tf.abs(true_features-pred_features))

def fft_loss(y_true,y_pred):

    y_true_fft=tf.signal.fft2d(tf.cast(y_true,tf.complex64))
    y_pred_fft=tf.signal.fft2d(tf.cast(y_pred,tf.complex64))

    return tf.reduce_mean(
        tf.abs(
            tf.abs(y_true_fft)-tf.abs(y_pred_fft)
        )
    )

# ================= UNIFIED TOTAL LOSS =================

def total_vae_loss(y_true,y_pred,z_mean,z_log_var,beta=4.0):

    recon_loss=reconstruction_loss(y_true,y_pred)

    kl_loss=kl_divergence_loss(z_mean,z_log_var)

    perc_loss=0.0
    freq_loss=0.0

    if USE_PERCEPTUAL_LOSS:
        perc_loss=perceptual_loss(y_true,y_pred)

    if USE_FFT_LOSS:
        freq_loss=fft_loss(y_true,y_pred)

    total_loss=(
        recon_loss+
        beta*kl_loss+
        0.1*perc_loss+
        0.1*freq_loss
    )

    return total_loss,recon_loss,kl_loss,perc_loss,freq_loss

# ================= VAE =================

class VAE(Model):

    def __init__(self,latent_dim):

        super(VAE,self).__init__()

        self.encoder=self.build_encoder(latent_dim)

        self.decoder=self.build_decoder(latent_dim)

    # ================= ENCODER =================

    def build_encoder(self,latent_dim):

        encoder_inputs=Input(shape=(128,128,3))

        x=Conv2D(32,3,activation='relu',strides=2,padding='same')(encoder_inputs)

        x=Conv2D(64,3,activation='relu',strides=2,padding='same')(x)

        x=Flatten()(x)

        x=Dense(256,activation='relu')(x)

        z_mean=Dense(latent_dim,name='z_mean')(x)

        z_log_var=Dense(latent_dim,name='z_log_var')(x)

        z=Sampling()([z_mean,z_log_var])

        return Model(
            encoder_inputs,
            [z_mean,z_log_var,z],
            name='Encoder'
        )

    # ================= DECODER =================

    def build_decoder(self,latent_dim):

        latent_inputs=Input(shape=(latent_dim,))

        x=Dense(32*32*64,activation='relu')(latent_inputs)

        x=Reshape((32,32,64))(x)

        x=Conv2DTranspose(64,3,activation='relu',strides=2,padding='same')(x)

        x=Conv2DTranspose(32,3,activation='relu',strides=2,padding='same')(x)

        decoder_outputs=Conv2DTranspose(
            3,3,
            activation='sigmoid',
            padding='same'
        )(x)

        return Model(
            latent_inputs,
            decoder_outputs,
            name='Decoder'
        )

    # ================= FORWARD =================

    def call(self,inputs):

        z_mean,z_log_var,z=self.encoder(inputs)

        reconstructed=self.decoder(z)

        return reconstructed,z_mean,z_log_var

# ================= TRAINER =================

class VAETrainer:

    def __init__(self):

        self.vae=VAE(LATENT_DIM)

        self.optimizer=tf.keras.optimizers.Adam(1e-4)

        self.train_losses=[]

    def train(self,train_dataset,epochs=80):

        for epoch in range(epochs):

            epoch_losses=[]

            print(f"\nEpoch {epoch+1}/{epochs}")

            for batch in train_dataset:

                with tf.GradientTape() as tape:

                    reconstructed,z_mean,z_log_var=self.vae(batch)

                    total_loss,recon_loss,kl_loss,perc_loss,freq_loss=total_vae_loss(
                        batch,
                        reconstructed,
                        z_mean,
                        z_log_var,
                        beta=BETA
                    )

                gradients=tape.gradient(
                    total_loss,
                    self.vae.trainable_variables
                )

                self.optimizer.apply_gradients(
                    zip(gradients,self.vae.trainable_variables)
                )

                epoch_losses.append(total_loss.numpy())

            avg_loss=np.mean(epoch_losses)

            self.train_losses.append(avg_loss)

            print(f"Total Loss: {avg_loss:.4f}")
            print(f"Reconstruction Loss: {recon_loss:.4f}")
            print(f"KL Loss: {kl_loss:.4f}")
            print(f"Perceptual Loss: {perc_loss:.4f}")
            print(f"FFT Loss: {freq_loss:.4f}")

        plt.plot(self.train_losses)

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")

        plt.show()

    # ================= CLASSIFICATION =================

    def compute_accuracy_and_confusion(self,x,y_true):

        z_mean,z_log_var,z=self.vae.encoder(x)

        reconstructed=self.vae.decoder(z).numpy()

        x_flat=reconstructed.reshape(
            (reconstructed.shape[0],-1)
        )

        clf=LogisticRegression(max_iter=1000)

        clf.fit(x_flat,y_true)

        y_pred=clf.predict(x_flat)

        accuracy=accuracy_score(y_true,y_pred)

        return accuracy,y_true,y_pred,clf,reconstructed

# ================= METRICS =================

def calculate_sensitivity_specificity(y_true,y_pred):

    cm=confusion_matrix(y_true,y_pred)

    TP=cm[1,1]
    TN=cm[0,0]
    FP=cm[0,1]
    FN=cm[1,0]

    sensitivity=TP/(TP+FN) if (TP+FN)>0 else 0

    specificity=TN/(TN+FP) if (TN+FP)>0 else 0

    return sensitivity,specificity

def calculate_metrics(y_true,y_pred):

    sensitivity,specificity=calculate_sensitivity_specificity(
        y_true,y_pred
    )

    return {
        'Accuracy':accuracy_score(y_true,y_pred),
        'F1 Score':f1_score(y_true,y_pred,average='weighted'),
        'Precision':precision_score(y_true,y_pred,average='weighted'),
        'Recall':recall_score(y_true,y_pred,average='weighted'),
        'Sensitivity':sensitivity,
        'Specificity':specificity
    }

# ================= PLOT SENS SPEC =================

def plot_fold_sens_spec(fold_metrics):

    sensitivities=[fm['Sensitivity'] for fm in fold_metrics]

    specificities=[fm['Specificity'] for fm in fold_metrics]

    folds=range(1,len(fold_metrics)+1)

    plt.plot(folds,sensitivities,marker='o',label='Sensitivity')

    plt.plot(folds,specificities,marker='s',label='Specificity')

    plt.xticks(folds)

    plt.xlabel('Fold')
    plt.ylabel('Score')

    plt.title('Sensitivity and Specificity')

    plt.ylim(0,1)

    plt.legend()

    plt.show()

# ================= TRAIN TEST VALIDATION =================

def cross_validation(x_all,y_all,class_names,epochs=80):

    x_train,x_temp,y_train,y_temp=train_test_split(
        x_all,
        y_all,
        test_size=0.2,
        random_state=42,
        stratify=y_all
    )

    x_val,x_test,y_val,y_test=train_test_split(
        x_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp
    )

    print("\n========== DATA SPLIT ==========")

    print(f"Training Samples   : {len(x_train)}")
    print(f"Validation Samples : {len(x_val)}")
    print(f"Testing Samples    : {len(x_test)}")

    train_dataset=tf.data.Dataset.from_tensor_slices(
        x_train
    ).shuffle(1000).batch(BATCH_SIZE)

    val_dataset=tf.data.Dataset.from_tensor_slices(
        x_val
    ).batch(BATCH_SIZE)

    trainer=VAETrainer()

    trainer.train(train_dataset,epochs=epochs)

    accuracy,y_true,y_pred,clf,reconstructed=trainer.compute_accuracy_and_confusion(
        x_test,
        y_test
    )

    metrics=calculate_metrics(y_true,y_pred)

    print('\n========== TEST METRICS ==========')

    for key,value in metrics.items():

        print(f"{key}: {value:.4f}")

    fold_metrics=[metrics]

    # ================= CONFUSION MATRIX =================

    cm=confusion_matrix(y_true,y_pred)

    disp=ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    disp.plot(cmap='Blues',colorbar=False)

    plt.title('Test Confusion Matrix')

    plt.show()

    # ================= ROC CURVE =================

    if hasattr(clf,"predict_proba"):

        y_score=clf.predict_proba(
            reconstructed.reshape(len(y_true),-1)
        )[:,1]

        fpr,tpr,_=roc_curve(y_true,y_score)

        roc_auc=auc(fpr,tpr)

        plt.plot(
            1-fpr,
            tpr,
            label='ROC (AUC = {:.2f})'.format(roc_auc)
        )

        plt.xlabel('Specificity')

        plt.ylabel('Sensitivity')

        plt.title('ROC Curve')

        plt.legend()

        plt.show()

    return fold_metrics

# ================= SUMMARY =================

def summarize_metrics(fold_metrics):

    avg_metrics={
        key:np.mean([
            fm[key]
            for fm in fold_metrics
        ])
        for key in fold_metrics[0].keys()
    }

    print('\n========== Average Metrics ==========')

    for k,v in avg_metrics.items():

        print(f"{k}: {v:.4f}")

# ================= MAIN =================

if __name__=="__main__":

    x_all,y_all,class_names=load_dataset(DATASET_PATH)

    print('\nStarting Training')

    fold_metrics=cross_validation(
        x_all,
        y_all,
        class_names,
        epochs=EPOCHS
    )

    summarize_metrics(fold_metrics)

    plot_fold_sens_spec(fold_metrics)
