import { ChevronDown, ChevronUp } from "lucide-react";
import { useEffect, useState } from "react";
import { Button } from "@mui/material";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import { styled } from "@mui/material/styles";
import React from 'react';
import { DotLottieReact } from '@lottiefiles/dotlottie-react';
const Home = ({ setPoke, pokeref }) => {
  const [d1, setD1] = useState(0);
  const [d2, setD2] = useState(0);
  const [d3, setD3] = useState(0);
  const [d4, setD4] = useState(0);
  const [d5, setD5] = useState(0);
  const [d6, setD6] = useState(0);
  const [jsonreturn, setjson] = useState();
  const result = ["Low", "Moderate", "High"];
  const [responeOnrender, setresponeOnrender] = useState(false);
  const [firstcome, setFisrtcome] = useState(false);
  const [isdropopen, setdropopen] = useState(false);
  const [model, setmodel] = useState(0);
  const [image, setImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [nnRespon, setNnrespon] = useState({ index: null, acc: null });
  const lable = [
    "Blazer",
    "Long Pants",
    "Shorts",
    "Dresses",
    "Hoodie",
    "Jacket",
    "Denim Jacket",
    "Sports Jacket",
    "Jeans",
    "T-Shirt",
    "Shirt",
    "Coat",
    "Polo",
    "Skirt",
    "Sweter",
  ];
  const VisuallyHiddenInput = styled("input")({
    clip: "rect(0 0 0 0)",
    clipPath: "inset(50%)",
    height: 1,
    overflow: "hidden",
    position: "absolute",
    bottom: 0,
    left: 0,
    whiteSpace: "nowrap",
    width: 1,
  });
  useEffect(() => {
    if (firstcome) {
      setPoke(true);
    }
  }, [firstcome]);
  useEffect(() => {
    if (!firstcome && !pokeref) {
      console.log("poke onrender!");
      poke();
      setFisrtcome(true);
    }
  }, []);
  const handleImageChange = (event) => {
    const file = event.target.files[0]; // รับไฟล์ที่เลือก
    if (file) {
      setImage(file); // เก็บไฟล์
      setImagePreview(URL.createObjectURL(file)); // สร้าง URL สำหรับแสดงตัวอย่าง
    }
  };

  // ฟังก์ชันสำหรับส่งภาพไปที่ API
  const handleSubmit = async (event) => {
    event.preventDefault();

    if (!image) {
      alert("Please select an image to upload.");
      return;
    }

    // แปลงไฟล์เป็น Base64
    const reader = new FileReader();
    reader.readAsDataURL(image); // แปลงเป็น Base64
    reader.onloadend = async () => {
      const base64Image = reader.result;

      // ส่งข้อมูลในรูปแบบ JSON
      const jsonData = {
        image: base64Image, // เพิ่มข้อมูล Base64
      };

      try {
        const response = await fetch("http://modelapi-3-flrg.onrender.com/predict_nn", {
          method: "POST",
          headers: {
            "Content-Type": "application/json", // บอกว่าเราส่งข้อมูลในรูปแบบ JSON
          },
          body: JSON.stringify(jsonData), // ส่งข้อมูลในรูปแบบ JSON
        });

        const result = await response.json();
        const max = Math.max(...result.prediction[0]);
        const indexmax = result.prediction[0].indexOf(max);
        setNnrespon({ index: indexmax, acc: max });
        console.log("Prediction result:", indexmax, " ", max);
      } catch (error) {
        console.error("Error uploading image:", error);
      }
    };
  };

  const poke = async () => {
    const res = await fetch(
      "https://modelapi-3-flrg.onrender.com/predict_knn",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          input: [0, 0, 0, 0, 0, 0],
        }),
      }
    );
    if (res.ok) {
      console.log("poked");
      setresponeOnrender(true);
    } else {
      console.log(res.status);
    }
    return { status: res.status };
  };
  const testapi = async () => {
    const ipdata = [d1, d2, d3, d4, d5, d6];
    console.log(ipdata);
    const min = Math.min(...ipdata);
    const max = Math.max(...ipdata);
    // สเกลข้อมูลที่ได้มา
    const scaledData = ipdata.map((item) =>
      item <= 0 ? 0 : (item - min) / (max - min)
    );
    console.log(scaledData);
    if (model === 0) {
      const res = await fetch(
        "https://modelapi-3-flrg.onrender.com/predict_knn",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            input: scaledData,
          }),
        }
      );
      const data = await res.json();
      console.log(model === 0 ? "knn: " : "svm: ", data);
      setjson(data?.prediction[0]);
      return { status: res.status, data };
    } else {
      const res = await fetch(
        "https://modelapi-3-flrg.onrender.com/predict_svm",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            input: scaledData,
          }),
        }
      );
      const data = await res.json();
      console.log(model === 0 ? "knn: " : "svm: ", data);
      setjson(data?.prediction[0]);
      return { status: res.status, data };
    }
  };
  return (
    <div className="flex flex-col items-center justify-center py-16 font-prompt pt-25 min-h-[100vh] ">
      <div className="flex flex-col md:flex-row justify-between w-full max-w-6xl px-4 md:px-0  relative">
        {/* Left Section */}
        <div className="text-center md:text-left max-w-xs mb-0 md:mb-32  h-[100%] flex-col min-w-[19vw] ">
          <div className="">
            <h1 className="text-xl font-semibold text-[#1F2F4D] " id="target">
              StressLv By Liftstyle
            </h1>
            <div
              className="select-none cursor-pointer rounded-[8px] px-4 py-2 border-1 border-[#1a1a1a2d] mt-2 ml-8"
              onClick={() => setdropopen(!isdropopen)}
            >
              <p className="flex justify-between">
                {model === 0 ? "K-Nearest Neighbors" : "Support Vector Model"}{" "}
                {isdropopen ? (
                  <ChevronUp className="ml-4" />
                ) : (
                  <ChevronDown className="ml-4" />
                )}
              </p>
              {isdropopen && (
                <div
                  className="
                flex-col items-center
                absolute bg-white w-[243px] mt-3 py-4 left-8 shadow-[0px_0px_2px_1px_#1a1a1a1d] rounded-[8px]"
                >
                  <p
                    onClick={() => setmodel(0)}
                    className="mb-3 w-full px-5 py-2 hover:bg-[#FF7F2C3d]"
                  >
                    K-Nearest Neighbors
                  </p>
                  <p
                    onClick={() => setmodel(1)}
                    className="mb-3 w-full px-5 py-2 hover:bg-[#FF7F2C3d]"
                  >
                    Support Vector Model
                  </p>
                </div>
              )}
            </div>
          </div>
          <div className="pl-8 pt-4">
            <label className="flex font-medium mb-1 items-end" htmlFor="study">
              <p className="text-[12px]">Study Time Hr / Day </p>
            </label>
            <input
              type="number"
              step="0.01"
              id="study"
              min="0"
              placeholder="ex. 0.06"
              className="w-full border border-[#1F2F4D] rounded px-3 py-2
              focus:outline-none focus:border-[#FF7F2C] focus:ring-1 focus:ring-[#FF7F2C]"
              onChange={(e) => setD1(e.target.value)}
            />
          </div>
          <div className="pt-2 pl-8">
            <label className="block font-medium mb-1" htmlFor="Extracurricular">
              <p className="text-[12px]">Extracurricular Time Hr / Day </p>
            </label>
            <input
              type="number"
              step="0.01"
              id="Extracurricular"
              min="0"
              placeholder="ex. 0.875"
              className="w-full border border-[#1F2F4D] rounded px-3 py-2
              focus:outline-none focus:border-[#FF7F2C] focus:ring-1 focus:ring-[#FF7F2C]"
              onChange={(e) => setD2(e.target.value)}
            />
          </div>
          <div className="pt-2 pl-8">
            <label className="block font-medium mb-1" htmlFor="Sleep">
              <p className="text-[12px]">Sleep Time Hr / Day </p>
            </label>
            <input
              type="number"
              step="0.01"
              id="Sleep"
              min="0"
              placeholder="ex. 0.60"
              className="w-full border border-[#1F2F4D] rounded px-3 py-2
              focus:outline-none focus:border-[#FF7F2C] focus:ring-1 focus:ring-[#FF7F2C]"
              onChange={(e) => setD3(e.target.value)}
            />
          </div>
          <div className="pt-2 pl-8">
            <label className="block font-medium mb-1" htmlFor="Social">
              <p className="text-[12px]">Social Time Hr / Day </p>
            </label>
            <input
              type="number"
              step="0.01"
              id="Social"
              min="0"
              placeholder="ex. 0.70"
              className="w-full border border-[#1F2F4D] rounded px-3 py-2
              focus:outline-none focus:border-[#FF7F2C] focus:ring-1 focus:ring-[#FF7F2C]"
              onChange={(e) => setD4(e.target.value)}
            />
          </div>
          <div className="pt-2 pl-8">
            <label className="block font-medium mb-1" htmlFor="Physical">
              <p className="text-[12px]">Physical Activity Time Hr / Day </p>
            </label>
            <input
              type="number"
              step="0.01"
              min="0"
              id="Physical"
              placeholder="ex. 0.27"
              className="w-full border border-[#1F2F4D] rounded px-3 py-2
              focus:outline-none focus:border-[#FF7F2C] focus:ring-1 focus:ring-[#FF7F2C]"
              onChange={(e) => setD5(e.target.value)}
            />
          </div>
          <div className="pt-2 pl-8">
            <label className="block font-medium mb-1" htmlFor="GPA">
              <p className="text-[12px]">GPA</p>
            </label>
            <input
              type="number"
              step="0.01"
              id="GPA"
              min="0"
              placeholder="ex. 0.28"
              className="w-full border border-[#1F2F4D] rounded px-3 py-2
              focus:outline-none focus:border-[#FF7F2C] focus:ring-1 focus:ring-[#FF7F2C]"
              onChange={(e) => setD6(e.target.value)}
            />
          </div>
          <div className="pl-8">
            <a onClick={testapi}>
              <span
                className="my-5 mx-5 bg-white text-[#20BEFF] py-2 px-4 w-50 text-center font-[400] border-1 justify-center
                    rounded-[5px] cursor-pointer shadow-md hover:bg-[#20BEFF] transition-bg ease-in-out duration-200 flex
                    hover:text-white transition-text hover:border-[#20BEFF]"
              >
                Predict Stress Level
              </span>
            </a>
            Predict: {result[jsonreturn]}
          </div>
        </div>

        {/* Illustration */}
        <div className="relative flex justify-center items-end w-64 h-64 md:w-96 md:h-150 md:my-0 ">
          <div className="absolute z-11 top-0 mt-10 flex-col flex items-center justify-center text-center ">
              {!responeOnrender || !firstcome &&  
              <DotLottieReact
              src="https://lottie.host/585383b0-1b84-4f36-bf0d-19e754a45eaa/h2IIMfQRQJ.lottie"
              loop
              autoplay
              className="w-30 -mt-10 "
            />}
            <h1 className="">
              {responeOnrender ||firstcome
                ? "Model Api is Ready!"
                : "Be Patiance invoking onrenderAPI..."}
            </h1>
            {!responeOnrender && (
              <a
                href="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
                target="_blank"
                rel="noopener noreferrer"
                className="text-[#FF7F2C] hover:underline"
              >
                while wait listen this!
              </a>
            )}
          </div>
          <img
            src={responeOnrender ? "/assets/smile.svg" : "/assets/nosmile.svg"}
            alt=""
            className="z-10 h-[80%]"
          />
          <div className="absolute w-60 h-60 md:w-84 md:h-84 bg-[#FF7F2C] rounded-full"></div>
        </div>

        {/* Right Section */}
        <div className="text-center md:text-right mt-0 md:mt-0  h-[100vh] min-w-[19vw] max-w-[19vw]">
          <h1 className="text-xl font-semibold text-[#1F2F4D] ">
            Clothes Classification
          </h1>
          <div className="flex justify-center mb-10  w-full min-h-[50%] min-w-[100%] max-w-[100%] mt-2">
            <div className=" fit px-10 border-1 rounded-[8px] border-dashed bor min-w-[100%] py-3">
              {imagePreview && (
                <img
                  src={imagePreview}
                  alt="Preview"
                  className=" rounded-[5px]"
                />
              )}
            </div>
          </div>
          <div className="">
            <Button
              component="label"
              role={undefined}
              tabIndex={-1}
              startIcon={<CloudUploadIcon />}
              sx={{
                backgroundColor: "white",
                color: "#FF7F2C",
                borderColor: "#1a1a1a1d",
                borderWidth: "1px",
                borderStyle: "solid",
              }}
            >
              Upload files
              <VisuallyHiddenInput
                type="file"
                onChange={handleImageChange}
                multiple
              />
            </Button>
          </div>
          <div className="pl-8">
            <a onClick={handleSubmit}>
              <span
                className="my-5 mx-5 bg-white text-[#20BEFF] py-2 px-4 w-50 text-center font-[400] border-1 justify-center
                    rounded-[5px] cursor-pointer shadow-md hover:bg-[#20BEFF] transition-bg ease-in-out duration-200 flex
                    hover:text-white transition-text hover:border-[#20BEFF]"
              >
                Predict Cloth
              </span>
            </a>
            <div className=" w-full">
              <p className="text-left">Predidt: {lable[nnRespon?.index]}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home;
