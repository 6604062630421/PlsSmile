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
const MLdata = () => {
  const [data, setData] = useState([]);
  useEffect(() => {
    fetch("src/assets/file/student_lifestyle_dataset.csv")
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
          src="src/assets/img/Learning.svg"
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
                      <TableCell key={i}>{cell}</TableCell>
                    ))}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </div>
        <div className="pt-5">
          <h1 className="text-2xl font-semibold text-[#3E4450ad] pl-5">
            Data Preprocessing
          </h1>
          <Copybox text="target = df['Stress_Level'].to_numpy().tolist()
stuhpd = df['Study_Hours_Per_Day'].to_numpy().tolist()
ehpd = df['Extracurricular_Hours_Per_Day'].to_numpy().tolist()
sleeppd = df['Sleep_Hours_Per_Day'].to_numpy().tolist()
socialpd = df['Social_Hours_Per_Day'].to_numpy().tolist()
papd = df['Physical_Activity_Hours_Per_Day'].to_numpy().tolist()
gpa = df['GPA'].to_numpy().tolist()
data = list(zip(stuhpd,ehpd,sleeppd,socialpd,papd,gpa))
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(data,target)" lang="python"/>
        </div>
      </div>
    </div>
  );
};

export default MLdata;
