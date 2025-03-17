import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from "@mui/material";
import Papa from "papaparse";
import React, { useState, useEffect } from "react";
import { Link as Scroll } from "react-scroll";
import Copybox from "../component/copybox";
import { ArrowUpRight } from "lucide-react";
const MLdata = () => {
  const [data, setData] = useState([]);
  const datatab = [
    {
      Study_Hours_Per_Day: 0.38,
      Extracurricular_Hours_Per_Day: 0.95,
      Sleep_Hours_Per_Day: 0.74,
      Social_Hours_Per_Day: 0.466667,
      Physical_Activity_Hours_Per_Day: 0.138462,
      GPA: 0.426136,
      Stress_Level: 1,
    },
    {
      Study_Hours_Per_Day: 0.06,
      Extracurricular_Hours_Per_Day: 0.875,
      Sleep_Hours_Per_Day: 0.6,
      Social_Hours_Per_Day: 0.7,
      Physical_Activity_Hours_Per_Day: 0.230769,
      GPA: 0.289773,
      Stress_Level: 0,
    },
    {
      Study_Hours_Per_Day: 0.02,
      Extracurricular_Hours_Per_Day: 0.975,
      Sleep_Hours_Per_Day: 0.84,
      Social_Hours_Per_Day: 0.2,
      Physical_Activity_Hours_Per_Day: 0.353846,
      GPA: 0.244318,
      Stress_Level: 0,
    },
    {
      Study_Hours_Per_Day: 0.3,
      Extracurricular_Hours_Per_Day: 0.525,
      Sleep_Hours_Per_Day: 0.44,
      Social_Hours_Per_Day: 0.283333,
      Physical_Activity_Hours_Per_Day: 0.5,
      GPA: 0.363636,
      Stress_Level: 1,
    },
    {
      Study_Hours_Per_Day: 0.62,
      Extracurricular_Hours_Per_Day: 0.15,
      Sleep_Hours_Per_Day: 0.3,
      Social_Hours_Per_Day: 0.366667,
      Physical_Activity_Hours_Per_Day: 0.507692,
      GPA: 0.721591,
      Stress_Level: 2,
    },
  ];
  useEffect(() => {
    fetch("/assets/student_lifestyle_dataset.csv")
      .then((respone) => respone.text())
      .then((csvText) => {
        Papa.parse(csvText, {
          complete: (res) => {
            setData(res.data);
          },
          header: true,
          skipEmptyLines: true,
        });
      })
      .catch((e) => console.log(e));
  }, []);
  return (
    <div className="flex flex-col px-50 pt-20 bg-white font-prompt min-h-[100vh] pb-30">
      <div className="pb-1 flex justify-center border-b-2 border-[#3E44502d]">
        <div className="flexcol w-[30%] pt-15 pb-10">
          <h1 className="font-semibold text-4xl text-[#1F2F4D] mb-2">
            Student's Stress level by Lifestyle
          </h1>
          <p className="flex text-1xl text-[#3E4450] mx-5">
            Machine Learning Model
          </p>
          <p className="flex text-1xl text-[#3E4450] mx-10">
            K-Nearest-Neighbors
          </p>
          <p className="flex text-1xl text-[#3E4450] mx-15">
            Support Vector Machine
          </p>
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
          src="/assets/Learning.svg"
          alt="Illustration"
          className="relative w-64 md:w-96 h-auto"
        />
      </div>
      <div className="pt-5 pb-1">
        <div className="px-10 mt-3">
          <p className="font-thin text-[15px]">
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This dataset, titled "Daily
            Lifestyle and Academic Performance of Students", contains data from
            2,000 students collected via a Google Form survey. It includes
            information on study hours, extracurricular activities, sleep,
            socializing, physical activity, stress levels, and CGPA. The data
            covers an academic year from August 2023 to May 2024 and reflects
            student lifestyles primarily from India. This dataset can help
            analyze the impact of daily habits on academic performance and
            student well-being.
          </p>
        </div>
      </div>
      <div>
      <h1 className="text-3xl font-semibold text-[#1F2F4D] " id="target">
          Dataset Referrence
        </h1>
        <a href="https://www.kaggle.com/datasets/steve1215rogg/student-lifestyle-dataset "target="_blank">
              <span
                className="my-5 mx-5 bg-white text-[#20BEFF] py-2 px-4 w-80 text-center font-[400] border-1 
                    rounded-lg cursor-pointer shadow-md hover:bg-[#20BEFF] transition-bg ease-in-out duration-200 flex justify-between
                    hover:text-white transition-text hover:border-[#20BEFF]"
              >
                {" "}
                Kaggle Student lifestyle dataset <ArrowUpRight/>
              </span>
            </a>
      </div>
      <div className="py-3">
        <h1 className="text-3xl font-semibold text-[#1F2F4D] " id="target">
          Dataset Overview
        </h1>
        <div className="px-10 mt-3 flex justify-center">
          <TableContainer sx={{ width: "80%" }}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell sx={{ fontWeight: "600", color: "#FF7F2C" }}>
                    ID
                  </TableCell>
                  <TableCell sx={{ fontWeight: "600", color: "#FF7F2C" }}>
                    Features
                  </TableCell>
                  <TableCell sx={{ fontWeight: "600", color: "#FF7F2C" }}>
                    Defination
                  </TableCell>
                  <TableCell sx={{ fontWeight: "600", color: "#FF7F2C" }}>
                    Data type
                  </TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                <TableRow>
                  <TableCell sx={{ fontWeight: "400", fontSize: "12px" }}>
                    1
                  </TableCell>
                  <TableCell sx={{ fontWeight: "400", fontSize: "12px" }}>
                    Study Hrs/Day
                  </TableCell>
                  <TableCell sx={{ fontWeight: "400", fontSize: "12px" }}>
                    Study hours of Student
                  </TableCell>
                  <TableCell sx={{ fontWeight: "400", fontSize: "12px" }}>
                    Float (hour / 24)
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell sx={{ fontWeight: "400", fontSize: "12px" }}>
                    2
                  </TableCell>
                  <TableCell sx={{ fontWeight: "400", fontSize: "12px" }}>
                    Extracurricular Hrs/Day
                  </TableCell>
                  <TableCell sx={{ fontWeight: "400", fontSize: "12px" }}>
                    Extracurricular study of Student
                  </TableCell>
                  <TableCell sx={{ fontWeight: "400", fontSize: "12px" }}>
                    Float (hour / 24)
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell sx={{ fontWeight: "400", fontSize: "12px" }}>
                    3
                  </TableCell>
                  <TableCell sx={{ fontWeight: "400", fontSize: "12px" }}>
                    Sleep Hrs/Day
                  </TableCell>
                  <TableCell sx={{ fontWeight: "400", fontSize: "12px" }}>
                    Sleep hours of Student
                  </TableCell>
                  <TableCell sx={{ fontWeight: "400", fontSize: "12px" }}>
                    Float (hour / 24)
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell sx={{ fontWeight: "400", fontSize: "12px" }}>
                    4
                  </TableCell>
                  <TableCell sx={{ fontWeight: "400", fontSize: "12px" }}>
                    Social Hrs/Day
                  </TableCell>
                  <TableCell sx={{ fontWeight: "400", fontSize: "12px" }}>
                    Social use of Student
                  </TableCell>
                  <TableCell sx={{ fontWeight: "400", fontSize: "12px" }}>
                    Float (hour / 24)
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell sx={{ fontWeight: "400", fontSize: "12px" }}>
                    5
                  </TableCell>
                  <TableCell sx={{ fontWeight: "400", fontSize: "12px" }}>
                    GPA
                  </TableCell>
                  <TableCell sx={{ fontWeight: "400", fontSize: "12px" }}>
                    GPA of Student
                  </TableCell>
                  <TableCell sx={{ fontWeight: "400", fontSize: "12px" }}>
                    Float (hour / 24)
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell sx={{ fontWeight: "400", fontSize: "12px" }}>
                    6
                  </TableCell>
                  <TableCell sx={{ fontWeight: "400", fontSize: "12px" }}>
                    Stress_Level
                  </TableCell>
                  <TableCell sx={{ fontWeight: "400", fontSize: "12px" }}>
                    Stress Level of Student
                  </TableCell>
                  <TableCell sx={{ fontWeight: "400", fontSize: "12px" }}>
                    Text (High,Moderate,Low)
                  </TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
        </div>
      </div>
      <div className="py-3">
        <h1 className="text-3xl font-semibold text-[#1F2F4D]">
          Exploratory Data Analysis (EDA)
        </h1>
        <h1 className="text-2xl font-semibold text-[#3E4450ad] pl-5 pt-4">
          Raw Data
        </h1>
        <div className="px-10 mt-5 flex justify-center h-[40vh]">
          <TableContainer
            sx={{
              borderRadius: "8px",
              borderWidth: "1px",
              borderColor: "#1a1a1a1d",
              width: "90%",
              overflowY: "auto",
              boxShadow: "none",
            }}
          >
            <Table stickyHeader>
              <TableHead>
                <TableRow>
                  <TableCell sx={{ fontWeight: "600", color: "#FF7F2C" }}>
                    ID
                  </TableCell>
                  <TableCell sx={{ fontWeight: "600", color: "#FF7F2C" }}>
                    Study Hr
                  </TableCell>
                  <TableCell sx={{ fontWeight: "600", color: "#FF7F2C" }}>
                    Extra Hr
                  </TableCell>
                  <TableCell sx={{ fontWeight: "600", color: "#FF7F2C" }}>
                    sleep Hr
                  </TableCell>
                  <TableCell sx={{ fontWeight: "600", color: "#FF7F2C" }}>
                    Social Hr
                  </TableCell>
                  <TableCell sx={{ fontWeight: "600", color: "#FF7F2C" }}>
                    Physical Hr
                  </TableCell>
                  <TableCell sx={{ fontWeight: "600", color: "#FF7F2C" }}>
                    GPA
                  </TableCell>
                  <TableCell sx={{ fontWeight: "600", color: "#FF7F2C" }}>
                    StressLv
                  </TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {data.map((row, index) => (
                  <TableRow key={index}>
                    {Object.values(row).map((cell, i) => (
                      <TableCell key={i} sx={{ fontSize: "12px" }}>
                        {cell}
                      </TableCell>
                    ))}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </div>
        <div className="pt-5 px-5">
          <h1 className="text-2xl font-semibold text-[#3E4450ad]">
            Data Preprocessing
          </h1>
          <div className="px-10 pt-3">
            <ul className="list-disc">
              <li className="text-[22px] font-medium text-[#FF7F2C]">
                Features Selection
              </li>
              <p>
                All label need to be in process but ID and pick Stress Level to
                output
              </p>
              <Copybox
                text="X = df.drop(['Student_ID','Stress_Level'],axis=1)
y = df['Stress_Level']"
                lang="python"
              />

              <div className="flex pt-10 ">
                <div className="w-1/2">
                  <li className="text-[22px] font-medium text-[#FF7F2C] pt-10">
                    Adjust & Scalling
                  </li>
                  <p>
                    Find a null data for Prevents Errors during training and
                    improve model performance
                  </p>
                  <Copybox text="df.isnull().any()" lang="python" />
                  <p className="pt-3">
                    fix stress level datatype from text to integer
                  </p>
                  <Copybox
                    text="df['Stress_Level'] = df['Stress_Level'].map({'High': 2, 'Moderate': 1,'Low':0})"
                    lang="python"
                  />
                </div>
                <div className="w-1/2 flex justify-center">
                  <table className="w-80 border-collapse">
                    <thead className="border border-[#1a1a1a2d] rounded-[8px]">
                      <tr>
                        <th className="text-left font-semibold text-[#FF7F2C] p-1">
                          Features
                        </th>
                        <th className="text-left font-semibold text-[#FF7F2C]"></th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr className="border border-[#1a1a1a2d]">
                        <td className="p-1 text-sm">Student_ID</td>
                        <td className="p-1 text-sm">False</td>
                      </tr>
                      <tr className="border border-[#1a1a1a2d]">
                        <td className="p-1 text-sm">Study_Hours_Per_Day</td>
                        <td className="p-1 text-sm">False</td>
                      </tr>
                      <tr className="border border-[#1a1a1a2d]">
                        <td className="p-1 text-sm">
                          Extracurricular_Hours_Per_Day
                        </td>
                        <td className="p-1 text-sm">False</td>
                      </tr>
                      <tr className="border border-[#1a1a1a2d]">
                        <td className="p-1 text-sm">Sleep_Hours_Per_Day</td>
                        <td className="p-1 text-sm">False</td>
                      </tr>
                      <tr className="border border-[#1a1a1a2d]">
                        <td className="p-1 text-sm">Social_Hours_Per_Day</td>
                        <td className="p-1 text-sm">False</td>
                      </tr>
                      <tr className="border border-[#1a1a1a2d]">
                        <td className="p-1 text-sm">
                          Physical_Activity_Hours_Per_Day
                        </td>
                        <td className="p-1 text-sm">False</td>
                      </tr>
                      <tr className="border border-[#1a1a1a2d]">
                        <td className="p-1 text-sm">GPA</td>
                        <td className="p-1 text-sm">False</td>
                      </tr>
                      <tr className="border border-[#1a1a1a2d]">
                        <td className="p-1 text-sm">Stress_Level</td>
                        <td className="p-1 text-sm">False</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
              <div className="pt-5">
                <p>Scalling Dataset</p>
                <Copybox
                  text="scaler = MinMaxScaler()
df[['Study_Hours_Per_Day','Extracurricular_Hours_Per_Day','Sleep_Hours_Per_Day','Social_Hours_Per_Day','Physical_Activity_Hours_Per_Day','GPA']] = scaler.fit_transform(df[['Study_Hours_Per_Day','Extracurricular_Hours_Per_Day','Sleep_Hours_Per_Day','Social_Hours_Per_Day','Physical_Activity_Hours_Per_Day','GPA']])"
                  lang="python"
                />
                <div className="pt-5 ">
                  <p className="pb-3">Example Data after Scaling</p>
                  <div className="w-full flex justify-center">
                    <TableContainer
                      sx={{
                        borderRadius: "8px",
                        borderWidth: "1px",
                        borderColor: "#1a1a1a1d",
                        width: "95%",
                        overflowY: "auto",
                        boxShadow: "none",
                      }}
                    >
                      {datatab && datatab.length > 0 ? (
                        <Table>
                          <TableHead>
                            <TableRow>
                              <TableCell
                                sx={{ fontWeight: "600", color: "#FF7F2C" }}
                              >
                                Study Hr
                              </TableCell>
                              <TableCell
                                sx={{ fontWeight: "600", color: "#FF7F2C" }}
                              >
                                Extra Hr
                              </TableCell>
                              <TableCell
                                sx={{ fontWeight: "600", color: "#FF7F2C" }}
                              >
                                sleep Hr
                              </TableCell>
                              <TableCell
                                sx={{ fontWeight: "600", color: "#FF7F2C" }}
                              >
                                Social Hr
                              </TableCell>
                              <TableCell
                                sx={{ fontWeight: "600", color: "#FF7F2C" }}
                              >
                                Physical Hr
                              </TableCell>
                              <TableCell
                                sx={{ fontWeight: "600", color: "#FF7F2C" }}
                              >
                                GPA
                              </TableCell>
                              <TableCell
                                sx={{ fontWeight: "600", color: "#FF7F2C" }}
                              >
                                StressLv
                              </TableCell>
                            </TableRow>
                          </TableHead>
                          <TableBody>
                            {datatab.map((row, index) => (
                              <TableRow key={index}>
                                {Object.values(row || {}).map((value, idx) => (
                                  <TableCell
                                    key={idx}
                                    sx={{ fontSize: "12px" }}
                                  >
                                    {value}
                                  </TableCell>
                                ))}
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      ) : (
                        <p>No data available</p>
                      )}
                    </TableContainer>
                  </div>
                </div>
              </div>
              <div>
                <li className="text-[22px] font-medium text-[#FF7F2C] pt-10">
                  Split Data
                </li>
                <p>Split data to training 70% and test 30%</p>
                <Copybox
                  text="from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)"
                  lang="python"
                />
              </div>
            </ul>
          </div>
        </div>
      </div>
      <div>
        <h1 className="text-3xl font-semibold text-[#1F2F4D]">Picked Model</h1>
        <div className="pl-5">
          <h1 className="text-2xl font-semibold text-[#3E4450ad]  pt-4">
            K-Nearest Neighbors ( KNN )
          </h1>
          <p>
            KNN stands for K-Nearest Neighbors, which is a simple and widely
            used machine learning algorithm for classification and regression
            tasks.
          </p>
          <p className="text-[#FF7F2C] pt-4">How KNN Works :</p>
          <ul className="list-disc ml-5">
            <li>
              It finds the K closest data points (neighbors) to a given input.
            </li>
            <li>
              Uses majority voting (for classification) or averaging (for
              regression) to predict the output.
            </li>
          </ul>
          <p className="text-[#FF7F2C] pt-4">Hyperparameter :</p>
          <ul className="list-disc ml-5">
            <Copybox text="n_neighbors" lang="python" />
            <li className="text-[#FF7F2C] ml-4 pt-3">K Parameters</li>
            <p className="ml-4 pt-2">
              The K hyperparameter in KNN sets the number of neighbors for
              predictions. Small K risks overfitting, while large K may
              underfit. Optimal K is found via cross-validation.
            </p>
          </ul>
          <p className="text-[#FF7F2C] pt-4">Model Detail :</p>
          <div className="flex items-center">
            <Copybox
              text="knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(data,target)"
              lang="python"
            />
            <p className="ml-4 pt-2">
              A straightforward algorithm that assigns a class based on the
              majority of its k-nearest neighbors. This model utilizes 4
              neighbors for classification.
            </p>
          </div>
        </div>
        <div className="pl-5">
          <h1 className="text-2xl font-semibold text-[#3E4450ad]  pt-10">
            Support Vector Models ( SVM )
          </h1>
          <p>
            SVM is a robust algorithm that determines the best hyperplane to
            divide data points into distinct classes, performing effectively on
            both linear and nonlinear datasets.
          </p>
          <p className="text-[#FF7F2C] pt-4">Hyperparameter :</p>
          <ul className="list-disc ml-5">
            <Copybox
              text="kernel='poly'
degree=3
C=1"
              lang="python"
            />
            <li className="text-[#FF7F2C] ml-4 pt-3">Kernel Parameters</li>
            <p className="ml-4 pt-2">
              Uses a polynomial function to map data into a higher-dimensional
              space.
            </p>
            <li className="text-[#FF7F2C] ml-4 pt-3">Degree Parameters</li>
            <p className="ml-4 pt-2">
              This controls the degree of the polynomial kernel. In this case, a
              cubic polynomial is used. A higher degree leads to a more flexible
              decision boundary, but it can also increase the risk of
              overfitting.
            </p>
            <li className="text-[#FF7F2C] ml-4 pt-3">C Parameters</li>
            <p className="ml-4 pt-2">
              This is the regularization parameter. A higher value of C gives a
              lower bias (more complex model), but it can also lead to
              overfitting. A lower value of C increases bias but reduces
              overfitting by making the model simpler.
            </p>
          </ul>
          <p className="text-[#FF7F2C] pt-4">Model Detail :</p>
          <div className="flex items-center">
            <Copybox
              text="poly = svm.SVC(kernel='poly',degree=3,C=1)
poly.fit(x_train,y_train)"
              lang="python"
            />
            <p className="ml-4 pt-2">
              A straightforward algorithm that assigns a class based on the
              majority of its k-nearest neighbors. This model utilizes 4
              neighbors for classification.
            </p>
          </div>
        </div>
      </div>
      <div>
        <h1 className="text-3xl font-semibold text-[#1F2F4D] pt-8">
          Model Evaluate
        </h1>
        <Copybox
          text="y_pred = knn.predict(x_test)
accrate = accuracy_score(y_test,y_pred)
prec = precision_score(y_test,y_pred,average='macro')
recall = recall_score(y_test ,y_pred ,average = 'macro')
f1 = f1_score(y_test,y_pred,average = 'macro')
conf = confusion_matrix(y_test,y_pred)

print(f'Acc: {accrate}')
print(f'Prec: {prec}')
print(f'Recall: {recall}')
print(f'F1: {f1}')
print('confusion M: ')
print(conf)"
          lang="python"
        />
        <h1 className="text-2xl font-semibold text-[#FF7F2C] pl-5 pt-4">
          KNN Evaluate
        </h1>
        <div className="flex">
            <div className="w-1/2">
            <Copybox
          text="Acc: 0.9633333333333334
Prec: 0.95474259701678
Recall: 0.9654947332196189
F1: 0.9599249008310448
confusion M: 
 [ 94   2   0]
 [  5 179   4]
 [  1  10 305]"
          lang="python"
        />
            </div>
            <div className="w-1/2">
                <img src="/assets/knn.png" alt="knn confusion matrix" />
            </div>
        </div>
        <h1 className="text-2xl font-semibold text-[#FF7F2C] pl-5 pt-10">
          SVM Evaluate
        </h1>
        <div className="flex">
            <div className="w-1/2">
            <Copybox
          text="precision : 0.9769862172521816
recall : 0.968909911123081
f1 : 0.9727911618418812
confusion :
 [ 90   4   2]
 [  1 184   3]
 [  1   2 313]"
          lang="python"
        />
            </div>
            <div className="w-1/2">
                <img src="/assets/svm.png" alt="svm confusion matrix" />
            </div>
        </div>
      </div>
    </div>
  );
};

export default MLdata;
