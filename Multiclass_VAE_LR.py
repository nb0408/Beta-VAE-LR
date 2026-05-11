import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img,img_to_array

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

from sklearn.metrics import (
    accuracy_score,confusion_matrix,ConfusionMatrixDisplay,
    f1_score,precision_score,recall_score,
    roc_auc_score,roc_curve,auc
)

# ================= CONFIG =================

DATASET_PATH="/content/dataset"

IMG_SIZE=(128,128)
LATENT_DIM=64
BATCH_SIZE=8
EPOCHS=30
BETA=4.0

USE_PERCEPTUAL_LOSS=True
USE_FFT_LOSS=True

# ================= GPU MEMORY =================

gpus=tf.config.experimental.list_physical_devices('GPU')

if gpus:

    try:

        for gpu in gpus:

            tf.config.experimental.set_memory_growth(gpu,True)

    except Exception as e:

        print("Could not set memory growth:",e)

# ================= DATASET =================

def load_dataset(dataset_path):

    images=[]
    labels=[]
    names=[]

    class_names=sorted([
        folder for folder in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path,folder))
    ])

    NUM_CLASSES=len(class_names)

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

                names.append(image_name)

            except:
                pass

    images=np.array(images,dtype=np.float32)

    labels=np.array(labels)

    names=np.array(names)

    print(f"\nTotal Images: {len(images)}")

    return images,labels,names,class_names,NUM_CLASSES

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

class VAE(tf.keras.Model):

    def __init__(self,latent_dim,beta=4.0):

        super().__init__()

        self.beta=beta

        self.encoder=self.build_encoder(latent_dim)

        self.decoder=self.build_decoder(latent_dim)

    # ================= ENCODER =================

    def build_encoder(self,latent_dim):

        inputs=Input(shape=(128,128,3))

        x=Conv2D(32,3,activation='relu',strides=2,padding='same')(inputs)

        x=Conv2D(64,3,activation='relu',strides=2,padding='same')(x)

        x=Flatten()(x)

        z_mean=Dense(latent_dim,name='z_mean')(x)

        z_log_var=Dense(latent_dim,name='z_log_var')(x)

        z=Sampling()([z_mean,z_log_var])

        return Model(inputs,[z_mean,z_log_var,z],name='Encoder')

    # ================= DECODER =================

    def build_decoder(self,latent_dim):

        latent_inputs=Input(shape=(latent_dim,))

        x=Dense(32*32*32,activation='relu')(latent_inputs)

        x=Reshape((32,32,32))(x)

        x=Conv2DTranspose(64,3,activation='relu',strides=2,padding='same')(x)

        x=Conv2DTranspose(32,3,activation='relu',strides=2,padding='same')(x)

        outputs=Conv2DTranspose(
            3,3,
            activation='sigmoid',
            padding='same'
        )(x)

        return Model(latent_inputs,outputs,name='Decoder')

    # ================= FORWARD =================

    def call(self,inputs):

        z_mean,z_log_var,z=self.encoder(inputs)

        reconstruction=self.decoder(z)

        return reconstruction,z_mean,z_log_var

# ================= TRAINER =================

class VAETrainer:

    def __init__(self,beta=4.0):

        self.beta=beta

        self.train_losses=[]

    # ================= TRAIN =================

    def train(self,train_ds,val_ds,epochs=30):

        self.vae=VAE(latent_dim=LATENT_DIM,beta=self.beta)

        optimizer=tf.keras.optimizers.Adam(1e-4)

        for epoch in range(epochs):

            epoch_losses=[]

            val_losses=[]

            print(f"\nEpoch {epoch+1}/{epochs}")

            # ================= TRAINING =================

            for batch in train_ds:

                with tf.GradientTape() as tape:

                    reconstruction,z_mean,z_log_var=self.vae(batch)

                    total_loss,recon_loss,kl_loss,perc_loss,freq_loss=total_vae_loss(
                        batch,
                        reconstruction,
                        z_mean,
                        z_log_var,
                        beta=self.beta
                    )

                grads=tape.gradient(
                    total_loss,
                    self.vae.trainable_weights
                )

                optimizer.apply_gradients(
                    zip(grads,self.vae.trainable_weights)
                )

                epoch_losses.append(total_loss.numpy())

            # ================= VALIDATION =================

            for val_batch in val_ds:

                val_recon,val_mean,val_log=self.vae(val_batch)

                val_total,_,_,_,_=total_vae_loss(
                    val_batch,
                    val_recon,
                    val_mean,
                    val_log,
                    beta=self.beta
                )

                val_losses.append(val_total.numpy())

            avg_loss=np.mean(epoch_losses)

            avg_val_loss=np.mean(val_losses)

            self.train_losses.append(avg_loss)

            print(f"Training Loss: {avg_loss:.4f}")
            print(f"Validation Loss: {avg_val_loss:.4f}")
            print(f"Reconstruction Loss: {recon_loss:.4f}")
            print(f"KL Loss: {kl_loss:.4f}")
            print(f"Perceptual Loss: {perc_loss:.4f}")
            print(f"FFT Loss: {freq_loss:.4f}")

        plt.plot(self.train_losses)

        plt.xlabel("Epoch")

        plt.ylabel("Loss")

        plt.title("Training Loss")

        plt.show()

    # ================= METRICS =================

    def compute_metrics(self,x,y_true,num_classes,batch_size=32):

        recons=[]

        for i in range(0,len(x),batch_size):

            xb=x[i:i+batch_size]

            _,_,z=self.vae.encoder(xb)

            recon=self.vae.decoder(z).numpy()

            recons.append(recon)

        recon=np.concatenate(recons,axis=0)

        x_flat=recon.reshape((recon.shape[0],-1))

        clf=LogisticRegression(
            max_iter=1000,
            multi_class='multinomial'
        )

        clf.fit(x_flat,y_true)

        y_pred=clf.predict(x_flat)

        y_prob=clf.predict_proba(x_flat)

        acc=accuracy_score(y_true,y_pred)

        f1=f1_score(y_true,y_pred,average='macro')

        precision=precision_score(y_true,y_pred,average='macro')

        recall=recall_score(y_true,y_pred,average='macro')

        y_true_bin=label_binarize(
            y_true,
            classes=list(range(num_classes))
        )

        auc_score=roc_auc_score(
            y_true_bin,
            y_prob,
            average='macro',
            multi_class='ovr'
        )

        cm=confusion_matrix(y_true,y_pred)

        specificity_list=[]

        for i in range(len(cm)):

            TP=cm[i,i]

            FP=cm[:,i].sum()-TP

            FN=cm[i,:].sum()-TP

            TN=cm.sum()-(TP+FP+FN)

            specificity=TN/(TN+FP+1e-7)

            specificity_list.append(specificity)

        specificity=np.mean(specificity_list)

        return (
            acc,
            y_pred,
            f1,
            precision,
            recall,
            recall,
            specificity,
            auc_score,
            y_prob
        )

# ================= MAIN =================

if __name__=="__main__":

    x_all,y_all,names_all,class_names,NUM_CLASSES=load_dataset(
        DATASET_PATH
    )

    # ================= SPLIT =================

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

    train_ds=tf.data.Dataset.from_tensor_slices(
        x_train
    ).batch(BATCH_SIZE)

    val_ds=tf.data.Dataset.from_tensor_slices(
        x_val
    ).batch(BATCH_SIZE)

    trainer=VAETrainer(beta=BETA)

    trainer.train(
        train_ds,
        val_ds,
        epochs=EPOCHS
    )

    # ================= TEST METRICS =================

    acc,y_pred,f1,prec,rec,sens,spec,auc_score,y_prob=trainer.compute_metrics(
        x_test,
        y_test,
        NUM_CLASSES
    )

    print("\n========== TEST METRICS ==========")

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"Sensitivity: {sens:.4f}")
    print(f"Specificity: {spec:.4f}")
    print(f"AUC: {auc_score:.4f}")

    # ================= CONFUSION MATRIX =================

    cm=confusion_matrix(y_test,y_pred)

    disp=ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    disp.plot(cmap='Blues')

    plt.title("Confusion Matrix")

    plt.show()

    # ================= ROC CURVE =================

    y_test_bin=label_binarize(
        y_test,
        classes=list(range(NUM_CLASSES))
    )

    for i in range(NUM_CLASSES):

        fpr,tpr,_=roc_curve(
            y_test_bin[:,i],
            y_prob[:,i]
        )

        roc_auc=auc(fpr,tpr)

        plt.plot(
            fpr,
            tpr,
            label=f'Class {i} AUC={roc_auc:.2f}'
        )

    plt.plot([0,1],[0,1],'k--')

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('ROC Curve')

    plt.legend()

    plt.show()