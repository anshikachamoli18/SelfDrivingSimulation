from utils import *
from sklearn.model_selection import train_test_split

#### STEP 1 Collection of Data
path = 'myData'
data = importDataInformation(path)

#### STEP 2 Visualization and distribution of Data
data = balanceData(data, False)

#### STEP 3 Extracting the Images Path and Steering Angles
imagesPath, steering = loadData(path, data)

#### STEP 4 Splitting into Training and Validation
xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steering, test_size=0.2, random_state=5)
print('Total training images:', len(xTrain))
print('Total validation images:', len(xVal))
print('Total training steering:', len(yTrain))
print('Total validation steering:', len(yVal))

#### STEP 5 Augmentation of Data

#### STEP 6 Preprocessing of Data

#### STEP 7

#### STEP 8
model = createModel()
model.summary()

#### STEP 9 Fitting Model
history = model.fit(batchGenerator(xTrain, yTrain, 100, 1), steps_per_epoch=300, epochs=10, validation_data=batchGenerator(xVal, yVal, 100, 0), validation_steps=200)

#### STEP 10
model.save('model.h5')
print("Model saved")

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.ylim([0, 1])
plt.title('Loss')
plt.xlabel('Epoch')
plt.show()
