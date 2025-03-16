import { useEffect, useState } from "react";
const Home = ({setPoke,pokeref}) => {
  const [d1, setD1] = useState(0);
  const [d2, setD2] = useState(0);
  const [d3, setD3] = useState(0);
  const [d4, setD4] = useState(0);
  const [d5, setD5] = useState(0);
  const [d6, setD6] = useState(0);
  const [jsonreturn, setjson] = useState();
  const result = ['Low','Moderate','High']
  const [responeOnrender,setresponeOnrender] = useState(false);
  const [firstcome,setFisrtcome] = useState(false);
  useEffect(()=>{
    if(firstcome){
        setPoke(true);
    }
  },[firstcome]);
  useEffect(()=>{
    if(!firstcome && !pokeref){
        console.log("poke onrender!")
        poke();
        setFisrtcome(true)
    }
  },[])
  const poke = async () =>{
    const res = await fetch(
        "https://modelapi-3-flrg.onrender.com/predict_knn",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            input: [0,0,0,0,0,0],
          }),
        }
      );
      if(res.ok){
        console.log("poked");
        setresponeOnrender(true)
        
      }
      else{
        console.log(res.status)
      }
      return { status: res.status};
  }
  const testapi = async () => {
    const ipdata=[d1,d2,d3,d4,d5,d6];
    console.log(ipdata);
    const min = Math.min(...ipdata);
    const max = Math.max(...ipdata);
    // สเกลข้อมูลที่ได้มา
    const scaledData = ipdata.map((item) => item<=0?(0):((item - min) / (max - min)));
    console.log(scaledData);
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
    console.log(data);
    setjson(data?.prediction[0])
    return { status: res.status, data };
  };
  return (
    <div className="flex flex-col items-center justify-center py-16 font-prompt pt-25 min-h-[100vh]">
      <div className="flex flex-col md:flex-row items-center justify-between w-full max-w-6xl px-4 md:px-0  relative">
        {/* Left Section */}
        <div className="text-center md:text-left max-w-xs mb-0 md:mb-32  h-[100%] flex-col">
          <div className="">
            <h1 className="text-xl font-semibold text-[#1F2F4D] " id="target">
              StressLv By Liftstyle
            </h1>
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
        <div className="relative flex justify-center items-center w-64 h-64 md:w-96 md:h-96 my-8 md:my-0 pr-8">
          <div className="absolute w-60 h-60 md:w-84 md:h-84 bg-[#FF7F2C] rounded-full"></div>
        </div>

        {/* Right Section */}
        <div className="text-center md:text-right max-w-xs mt-0 md:mt-32 pr-8">
          <h3 className="text-xl font-bold text-gray-800"></h3>
          <p className="text-xl text-gray-600">
            
            
          </p>
          <a className="mt-4 inline-block bg-[#FF7F2C] text-white py-2 px-4 rounded-lg cursor-pointer shadow-md hover:shadow-[0px_0px_5px_2px_#FF7E697D] transition-shadow ease-in-out duration-200">
            
          </a>
        </div>
      </div>
    </div>
  );
};

export default Home;
