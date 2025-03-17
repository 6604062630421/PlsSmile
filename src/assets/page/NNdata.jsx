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
        <h1 className="text-2xl font-semibold text-[#3E4450ad]  pt-4 ml-10" >
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
      </div>
      <div className="py-3 ml-10">
      <h1 className="text-2xl font-semibold text-[#3E4450ad]  pt-4">
          Model fit and Hyperparameters setting
        </h1>
        <ul className="list-disc pt-6 ml-10">
              <li className="text-[18px] font-medium gap-2 text-[#FF7F2C]">
                Call back
              </li>
              <p>asd</p>
              <Copybox text="callbacks = [
    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-6),
    keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
]"/>
            </ul>
      </div>
      <div className="py-3 ml-10">
      <h1 className="text-2xl font-semibold text-[#3E4450ad]  pt-4">
          Model Evaluate
        </h1>
      </div>
    </div>
  );
};

export default NNdata;
