import React from "react";
import { Link as Scroll } from "react-scroll";
import { ArrowUpRight, Copy } from "lucide-react";
import Copybox from "../component/copybox";
const NNdata = () => {
  const exampledata = [
    "/assets/example/image (1).jpg",
    "/assets/example/image (2).jpg",
    "/assets/example/image (3).jpg",
    "/assets/example/image (4).jpg",
    "/assets/example/image (5).jpg",
    "/assets/example/image (6).jpg",
    "/assets/example/image (7).jpg",
    "/assets/example/image (8).jpg",
    "/assets/example/image (9).jpg",
    "/assets/example/image (10).jpg",
    "/assets/example/image (11).jpg",
    "/assets/example/image (12).jpg",
    "/assets/example/image (13).jpg",
  ];
  return (
    <div className="flex flex-col px-50 pt-20 bg-white font-prompt min-h-[100vh] pb-30">
      <div className="pb-1 flex justify-center border-b-2 border-[#3E44502d]">
        <div className="flexcol w-[30%] pt-15 pb-10">
          <h1 className="font-semibold text-4xl text-[#1F2F4D] mb-2">
            Clothes Classification
          </h1>
          <p className="flex text-1xl text-[#3E4450] mx-5">
            Neural Network Model
          </p>
          <p className="flex text-1xl text-[#3E4450] mx-10">
            Convolutional Neural Network
          </p>
          <p className="flex text-1xl text-[#3E4450] mx-15">&nbsp;</p>
          <div className=" fit justify-center flex">
            <Scroll to="target" smooth={true} duration={500} offset={-150}>
              <span
                className="mt-15 inline-block bg-[#FF7F2C] text-white py-2 px-4 w-80 text-center font-[600]
                                rounded-lg cursor-pointer shadow-md hover:shadow-[0px_0px_5px_2px_#FF7F2C4D] transition-shadow ease-in-out duration-200"
              >
                {" "}
                Explore Dataset
              </span>
            </Scroll>
          </div>
        </div>
        <img
          src="/assets/clothes.svg"
          alt="Illustration"
          className="relative w-64 md:w-96 h-auto"
        />
      </div>
      <div className="pt-5 pb-1">
        <div className="px-10 mt-3">
          <p className="font-thin text-[15px]">
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This dataset contains images of
            clothing items scraped from Carousell, an online marketplace,
            specifically curated for image classification tasks. It includes a
            diverse set of classes representing different types of clothing,
            making it an excellent resource for machine learning and computer
            vision projects. The images in this dataset represent various
            styles, textures, and colors, offering a comprehensive resource for
            training models to recognize and classify clothing categories. It is
            ideal for tasks such as building fashion recommendation systems,
            creating virtual try-on applications, or studying visual trends in
            fashion e-commerce. Whether you are an enthusiast or a professional,
            this dataset can help explore and experiment with deep learning
            techniques in the realm of fashion.
          </p>
        </div>
      </div>
      <div>
        <h1 className="text-3xl font-semibold text-[#1F2F4D] " id="target">
          Dataset Referrence
        </h1>
        <a
          href="https://www.kaggle.com/datasets/ryanbadai/clothes-dataset "
          target="_blank"
        >
          <span
            className="my-5 mx-5 bg-white text-[#20BEFF] py-2 px-4 w-80 text-center font-[400] border-1 
                    rounded-lg cursor-pointer shadow-md hover:bg-[#20BEFF] transition-bg ease-in-out duration-200 flex justify-between
                    hover:text-white transition-text hover:border-[#20BEFF]"
          >
            {" "}
            Clothes Dataset <ArrowUpRight />
          </span>
        </a>
      </div>
      <div className="py-3">
        <h1 className="text-3xl font-semibold text-[#1F2F4D] " id="target">
          Dataset Overview
        </h1>
        <div className="px-10 mt-3 justify-center">
          dataset consisting of images categorized into 15 classes, with each
          class containing 500 images. The classes are:
          <p>result : 7,500 picture</p>
          <ul className="list-disc pl-10 text-[#FF7F2C] pb-5">
            <li>Blazer</li>
            <li>Celana Panjang (Long Pants)</li>
            <li>Celana Pendek (Shorts)</li>
            <li>Gaun (Dresses)</li>
            <li>Hoodie</li>
            <li>Jaket (Jacket)</li>
            <li>Jaket Denim (Denim Jacket)</li>
            <li>Jaket Olahraga (Sports Jacket)</li>
            <li>Jeans</li>
            <li>Kaos (T-shirt)</li>
            <li>Kemeja (Shirt)</li>
            <li>Mantel (Coat)</li>
            <li>Polo</li>
            <li>Rok (Skirt)</li>
            <li>Sweter (Sweater)</li>
          </ul>
          <img src="/assets/datasetfolder.png" alt="datasetfolder" />
          <h1 className="text-2xl font-semibold text-[#3E4450ad] pl-5 pt-4 mb-3 ">
            Example Data
          </h1>
          <div className="justify-center flex flex-wrap gap-4 p-5 border-1 rounded-[8px] border-[#3E44502d]">
            {exampledata.map((data, index) => (
              <img
                src={data}
                alt={`example${index}`}
                key={index}
                className="w-25 h-25 object-cover rounded-[5px]"
              />
            ))}
          </div>
        </div>
      </div>
      <div>
        <h1 className="text-3xl font-semibold text-[#1F2F4D] " id="target">
          Dataset Preprocessing
        </h1>
        <div className="ml-10">
          <h1 className="text-2xl font-semibold text-[#3E4450ad]  pt-4">
            Import Dataset
          </h1>
          <p>import data from kagglehub api</p>
          <Copybox
            lang={"python"}
            text='import kagglehub

# Download latest version
path = kagglehub.dataset_download("ryanbadai/clothes-dataset")

print("Path to dataset files:", path)

train_dir = "/root/.cache/kagglehub/datasets/ryanbadai/clothes-dataset/versions/1/Clothes_Dataset"'
          />
          <h1 className="text-2xl font-semibold text-[#3E4450ad]  pt-4">
            Scaling Dataset
          </h1>
          <p>
            The code normalizes image data by scaling pixel values from [0, 255]
            to [0, 1] using Rescaling(1./255). This transformation is applied to
            both training and validation images while keeping the labels
            unchanged, improving model performance.
          </p>
          <Copybox
            lang={"python"}
            text="normalization_layer = tf.keras.layers.Rescaling(1./255)
train_data = train_data.map(lambda x, y: (normalization_layer(x), y))
val_data = val_data.map(lambda x, y: (normalization_layer(x), y))"
          />
          <h1 className="text-2xl font-semibold text-[#3E4450ad]  pt-4">
            Split Dataset
          </h1>
          <p>Split Dataset to Train data and Validate data</p>
          <div className="flex gap-18">
            <div className="w-1/2">
              <Copybox
                lang={"python"}
                text='train_data = image_dataset_from_directory(
    train_dir,
    image_size=(224, 224),
    batch_size=64,
    validation_split=0.2,
    subset="training",
    seed=123,
    shuffle=True,
    label_mode="categorical"  

)

val_data = image_dataset_from_directory(
    train_dir,
    image_size=(224, 224),
    batch_size=64,
    validation_split=0.2,
    subset="validation",
    seed=123,
    shuffle=True,
    label_mode="categorical"  

)'
              />
            </div>

            <div className="w-1/2">
              <ul className="list-disc pt-6">
                <li className="text-[13px] font-medium gap-2 text-[#FF7F2C]">
                  {" "}
                  image_size=(224, 224) →{" "}
                  <p className="text-[#1F2F4D]">
                    Resizes images to 224x224 pixels.
                  </p>
                </li>
                <li className="text-[13px] font-medium gap-2 text-[#FF7F2C]">
                  {" "}
                  batch_size=64 →{" "}
                  <p className="text-[#1F2F4D]">Loads 64 images per batch.</p>
                </li>
                <li className="text-[13px] font-medium gap-2 text-[#FF7F2C]">
                  {" "}
                  validation_split=0.2 →{" "}
                  <p className="text-[#1F2F4D]">
                    Uses 20% of the data for validation.
                  </p>
                </li>
                <li className="text-[13px] font-medium gap-2 text-[#FF7F2C]">
                  {" "}
                  subset="validation →{" "}
                  <p className="text-[#1F2F4D]">
                    Loads only the validation dataset
                  </p>
                </li>
                <li className="text-[13px] font-medium gap-2 text-[#FF7F2C]">
                  {" "}
                  seed=12 →{" "}
                  <p className="text-[#1F2F4D]">
                    Sets a seed to ensure consistent data splitting.
                  </p>
                </li>
                <li className="text-[13px] font-medium gap-2 text-[#FF7F2C]">
                  {" "}
                  shuffle=True →{" "}
                  <p className="text-[#1F2F4D]">
                    Randomizes the order of images before training.
                  </p>
                </li>
                <li className="text-[13px] font-medium gap-2 text-[#FF7F2C]">
                  {" "}
                  label_mode="categorical" →{" "}
                  <p className="text-[#1F2F4D]">
                    Converts labels to one-hot encoding (for multi-class
                    classification).
                  </p>
                </li>
              </ul>
            </div>
          </div>
        </div>
        <div className="ml-10">
          <h1 className="text-2xl font-semibold text-[#3E4450ad]  pt-4">
            Data Autotune
          </h1>
          <p>
            AUTOTUNE in TensorFlow is a feature that helps optimize data loading
            by automatically adjusting the buffer size for efficient
            performance. When used with
          </p>
          <Copybox
            lang={"python"}
            text="AUTOTUNE = tf.data.experimental.AUTOTUNE
train_data = train_data.prefetch(buffer_size=AUTOTUNE)
val_data = val_data.prefetch(buffer_size=AUTOTUNE)"
          />
          <ul className="list-disc pt-6">
            <li className="text-[13px] font-medium gap-2 text-[#FF7F2C]">
              {" "}
              AUTOTUNE →{" "}
              <p className="text-[#1F2F4D]">
                Automatically selects optimal buffer size
              </p>
            </li>
            <li className="text-[13px] font-medium gap-2 text-[#FF7F2C]">
              {" "}
              prefetch(AUTOTUNE) →{" "}
              <p className="text-[#1F2F4D]">
                Loads next batch while GPU is training
              </p>
            </li>
          </ul>
          <h1 className="text-2xl font-semibold text-[#3E4450ad]  pt-4">
            Data Augmentation
          </h1>
          <p>
            Data Augmentation is a technique used in machine learning,
            especially in deep learning and computer vision, to artificially
            increase the size and diversity of a dataset by applying
            transformations to existing data.
          </p>
          <div className="flex gap-18">
            <div className="w-1/2">
              <Copybox
                lang={"python"}
                text="from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    rescale=1./255
)

train_data_augmented = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    class_mode='categorical',
    subset='training',
    seed=123,
    shuffle=True
)"
              />
            </div>
            <div className="w-1/2">
              <ul className="list-disc pt-6">
                <li className="text-[13px] font-medium gap-2 text-[#FF7F2C]">
                  {" "}
                  rotation_range=20 →{" "}
                  <p className="text-[#1F2F4D]">Rotates image ±20°</p>
                </li>
                <li className="text-[13px] font-medium gap-2 text-[#FF7F2C]">
                  {" "}
                  width_shift_range=0.2 →{" "}
                  <p className="text-[#1F2F4D]">
                    Shifts image horizontally by 20%
                  </p>
                </li>
                <li className="text-[13px] font-medium gap-2 text-[#FF7F2C]">
                  {" "}
                  height_shift_range=0.2 →{" "}
                  <p className="text-[#1F2F4D]">
                    Shifts image vertically by 20%
                  </p>
                </li>
                <li className="text-[13px] font-medium gap-2 text-[#FF7F2C]">
                  {" "}
                  shear_range=0.2 →{" "}
                  <p className="text-[#1F2F4D]">
                    Applies shearing (skewing effect)
                  </p>
                </li>
                <li className="text-[13px] font-medium gap-2 text-[#FF7F2C]">
                  {" "}
                  zoom_range=0.2 →
                  <p className="text-[#1F2F4D]"> Zooms in/out by 20%</p>
                </li>
                <li className="text-[13px] font-medium gap-2 text-[#FF7F2C]">
                  {" "}
                  horizontal_flip=True →
                  <p className="text-[#1F2F4D]"> Flips image horizontally</p>
                </li>
                <li className="text-[13px] font-medium gap-2 text-[#FF7F2C]">
                  {" "}
                  fill_mode='nearest' →{" "}
                  <p className="text-[#1F2F4D]">
                    Fills missing pixels using nearest values
                  </p>
                </li>
                <li className="text-[13px] font-medium gap-2 text-[#FF7F2C]">
                  {" "}
                  rescale=1./255 →{" "}
                  <p className="text-[#1F2F4D]">
                    {" "}
                    Normalizes pixel values to [0,1]
                  </p>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>
      <div className="py-3">
        <h1 className="text-3xl font-semibold text-[#1F2F4D] ">
          Model Architecture
        </h1>
        <h1 className="text-2xl font-semibold text-[#3E4450ad]  pt-4 ml-10">
          Learning Transfer
        </h1>
        <p className="ml-10">
          Transfer Learning is a machine learning technique where a pre-trained
          model (trained on a large dataset) is used as a starting point for a
          new task. Instead of training a model from scratch
        </p>
        <div className="flex gap-15 ml-10">
          <div className="w-1/2">
            <Copybox
              lang={"python"}
              text="base_model = keras.applications.EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)


base_model.trainable = False

model = keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(15, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)"
            />
          </div>
          <div className="w-1/2">
            <ul className="list-disc pt-6">
              <li className="text-[13px] font-medium gap-2 text-[#FF7F2C]">
                EfficientNetB0 →
                <p className="text-[#1F2F4D]">
                  A deep learning model designed for image classification with
                  efficient scaling of width, depth, and resolution.
                </p>
              </li>
              <li className="text-[13px] font-medium gap-2 text-[#FF7F2C]">
                Compound Scaling →
                <p className="text-[#1F2F4D]">
                  Balances model width, depth, and resolution to optimize
                  performance.
                </p>
              </li>
              <li className="text-[13px] font-medium gap-2 text-[#FF7F2C]">
                Depthwise Separable Convolutions →
                <p className="text-[#1F2F4D]">
                  Improves model efficiency by reducing computational cost while
                  maintaining accuracy.
                </p>
              </li>
              <li className="text-[13px] font-medium gap-2 text-[#FF7F2C]">
                Squeeze-and-Excitation Blocks →
                <p className="text-[#1F2F4D]">
                  Focuses the model on important features by recalibrating
                  channel-wise feature responses.
                </p>
              </li>
              <li className="text-[13px] font-medium gap-2 text-[#FF7F2C]">
                Swish Activation →
                <p className="text-[#1F2F4D]">
                  Uses the Swish activation function for improved gradient flow
                  and better model training.
                </p>
              </li>
            </ul>
          </div>
        </div>
      </div>
      <div className="py-3 ml-10">
        <h1 className="text-2xl font-semibold text-[#3E4450ad]  pt-4">
          Layer Design
        </h1>
        <div className="flex gap-15">
          <div className="w-1/2">
            <Copybox
              text="from keras import regularizers
model = keras.models.Sequential()
model.add(keras.layers.InputLayer(shape=(224, 224, 3)))
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.3))

# Block 2
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.3))

# Block 3
model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.3))

# Block 4
model.add(keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.3))

# Fully connected layers
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(15, activation='softmax'))"
            />
            
          </div>
          <div className="w-1/2">
            <ul className="list-disc pt-6">
              <li className="text-[13px] font-medium gap-2 text-[#FF7F2C]">
                Input Layer →
                <p className="text-[#1F2F4D]">
                  Defines the input shape as (224, 224, 3) for image data.
                </p>
              </li>
              <li className="text-[13px] font-medium gap-2 text-[#FF7F2C]">
                Convolutional Blocks →
                <p className="text-[#1F2F4D]">
                  Consists of four blocks, each containing two Conv2D layers
                  (ReLU activation), MaxPooling2D, BatchNormalization, and
                  Dropout.
                </p>
              </li>
              <li className="text-[13px] font-medium gap-2 text-[#FF7F2C]">
                Fully Connected Layers →
                <p className="text-[#1F2F4D]">
                  Flattens the feature maps and passes them through Dense layers
                  with ReLU activation, L2 regularization, BatchNormalization,
                  and Dropout.
                </p>
              </li>
              <li className="text-[13px] font-medium gap-2 text-[#FF7F2C]">
                Output Layer →
                <p className="text-[#1F2F4D]">
                  Uses a Dense layer with 15 output neurons and softmax
                  activation for multi-class classification.
                </p>
              </li>
              <li className="text-[13px] font-medium gap-2 text-[#FF7F2C]">
                Conv2D →
                <p className="text-[#1F2F4D]">
                  Applies 128 convolution filters of size (3,3) with ReLU
                  activation and 'same' padding to preserve spatial dimensions.
                </p>
              </li>
              <li className="text-[13px] font-medium gap-2 text-[#FF7F2C]">
                MaxPooling2D →
                <p className="text-[#1F2F4D]">
                  Reduces spatial dimensions by half using a (2,2) pooling
                  window, retaining essential features while reducing
                  computation.
                </p>
              </li>
              <li className="text-[13px] font-medium gap-2 text-[#FF7F2C]">
                BatchNormalization →
                <p className="text-[#1F2F4D]">
                  Normalizes activations to stabilize training, improve
                  convergence speed, and reduce internal covariate shift.
                </p>
              </li>
              <li className="text-[13px] font-medium gap-2 text-[#FF7F2C]">
                Dropout →
                <p className="text-[#1F2F4D]">
                  Randomly disables 30% of neurons during training to prevent
                  overfitting and improve generalization.
                </p>
              </li>
              <li className="text-[13px] font-medium gap-2 text-[#FF7F2C]">
                Kernel Regularization (L2) →
                <p className="text-[#1F2F4D]">
                  Applies L2 regularization (λ=0.001) to Conv2D layers to reduce
                  overfitting by penalizing large weight values.
                </p>
              </li>
            </ul>
          </div>
        </div>
      </div>
      <div className="flex ml-10 gap-15">
        <div className="w-1/2"><img src="/assets/paramtab.png" alt="paramtab" className="max-w-[50vh] ml-20"/></div>
        <div className="w-1/2">
        <ul className="list-disc pt-6 pl-10">
            <li className="text-[13px] font-medium gap-2 text-[#FF7F2C]">
            Total params →
              <p className="text-[#1F2F4D]">
              27,003,055 (103.01 MB)
              </p>
            </li>
            <li className="text-[13px] font-medium gap-2 text-[#FF7F2C]">
            Trainable params →
              <p className="text-[#1F2F4D]">
              27,000,559 (103.00 MB)
              </p>
            </li>
            <li className="text-[13px] font-medium gap-2 text-[#FF7F2C]">
            Non-trainable paramss →
              <p className="text-[#1F2F4D]">
              2,496 (9.75 KB)
              </p>
            </li>
            </ul></div>
      </div>
      <div className="py-3 ml-10">
        <h1 className="text-2xl font-semibold text-[#3E4450ad]  pt-4">
          Model fit and Hyperparameters setting
        </h1>
        <ul className="list-disc pt-6 ml-10">
          <li className="text-[18px] font-medium gap-2 text-[#FF7F2C]">
            Model Complier
          </li>
          <Copybox
            text="optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, 
loss='categorical_crossentropy', 
metrics=['accuracy'])"
          />
          <ul className="list-disc pt-6 pl-10">
            <li className="text-[13px] font-medium gap-2 text-[#FF7F2C]">
              Adam Optimizer →
              <p className="text-[#1F2F4D]">
                Uses the Adam optimization algorithm with a learning rate of
                0.0001 for efficient weight updates.
              </p>
            </li>
            <li className="text-[13px] font-medium gap-2 text-[#FF7F2C]">
              Loss Function →
              <p className="text-[#1F2F4D]">
                Categorical Crossentropy is used as the loss function, suitable
                for multi-class classification tasks.
              </p>
            </li>
            <li className="text-[13px] font-medium gap-2 text-[#FF7F2C]">
              Metrics →
              <p className="text-[#1F2F4D]">
                Tracks accuracy during training to evaluate model performance.
              </p>
            </li>
          </ul>
          <li className="text-[18px] font-medium gap-2 text-[#FF7F2C]">
            Call back
          </li>
          <p>
            A callback in Keras is a function that runs during training to
            monitor and control the process, such as stopping early, adjusting
            learning rates, or saving the model.
          </p>
          <Copybox
            text="callbacks = [
    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-6),
    keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
]"
          />

          <ul className="list-disc pt-6 pl-10">
            <li className="text-[13px] font-medium gap-2 text-[#FF7F2C]">
              EarlyStopping →
              <p className="text-[#1F2F4D]">
                Stops training if validation loss doesn't improve for 10 epochs,
                restoring the best weights.
              </p>
            </li>
            <li className="text-[13px] font-medium gap-2 text-[#FF7F2C]">
              ReduceLROnPlateau →
              <p className="text-[#1F2F4D]">
                Reduces learning rate by a factor of 0.1 if validation loss
                stagnates for 5 epochs, with a minimum learning rate of 1e-6.
              </p>
            </li>
            <li className="text-[13px] font-medium gap-2 text-[#FF7F2C]">
              ModelCheckpoint →
              <p className="text-[#1F2F4D]">
                Saves the model as 'best_model.h5' whenever validation
                performance improves.
              </p>
            </li>
          </ul>
          <li className="text-[18px] font-medium gap-2 text-[#FF7F2C]">
            Model fit
          </li>
          <Copybox text="history = model.fit(train_data_augmented,epochs=100,batch_size=64, validation_data=val_data ,callbacks=[callbacks])" />
          <ul className="list-disc pt-6">
            <li className="text-[13px] font-medium gap-2 text-[#FF7F2C]">
              Training Process →
              <p className="text-[#1F2F4D]">
                Trains the model using the augmented training data for 100
                epochs with a batch size of 64.
              </p>
            </li>
            <li className="text-[13px] font-medium gap-2 text-[#FF7F2C]">
              Validation Data →
              <p className="text-[#1F2F4D]">
                Evaluates model performance on validation data during training.
              </p>
            </li>
            <li className="text-[13px] font-medium gap-2 text-[#FF7F2C]">
              Callbacks →
              <p className="text-[#1F2F4D]">
                Uses predefined callbacks to monitor training, adjust learning
                rate, and save the best model. As mentioned above
              </p>
            </li>
          </ul>
          <img
            src="/assets/exampletrain.png"
            alt="exampletrain"
            className="rounded-[5px] mt-3"
          />
        </ul>
      </div>
      <div className="py-3 ml-10">
        <h1 className="text-2xl font-semibold text-[#3E4450ad]  pt-4">
          Model Evaluate
        </h1> 
        Confusion Matrix (at 50 epcoh)
        <Copybox text='import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Predict on validation data
y_true = np.concatenate([y for x, y in val_data], axis=0)
y_pred_probs = model.predict(val_data)
y_pred = np.argmax(y_pred_probs, axis=1)

# Compute confusion matrix
conf_matrix = confusion_matrix(y_true.argmax(axis=1), y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels= class_names, yticklabels= class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()'/>
<img src="/assets/nnmatrix.png" alt="" />
Accurate & Loss Graph (at 50 epcoh)
<Copybox text="# Print classification report
print(classification_report(y_true.argmax(axis=1), y_pred, target_names=class_names))

plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()

plt.show()"/>
<img src="/assets/nngraph.png" alt="" />
      </div>
      
    </div>
  );
};

export default NNdata;
