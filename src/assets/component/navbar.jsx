import { useState, useEffect } from "react";
import { Link, useNavigate, useLocation } from "react-router";
import { X, Menu } from "lucide-react";
import styles from "./animate.module.css";
const Navbar = () => {
  const [isScrolled, setIsScrolled] = useState(false);
  const [isOpen, setIsOpen] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();
  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 0);
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);
  return (
    <nav
      className={`fixed top-0 left-0 w-full transition-all duration-300 z-10 font-prompt ${
        isScrolled ? "opacity-95" : "p-3"
      }`}
    >
      <div className="bg-white shadow-[0px_0px_2px_0px_#1a1a1a3d] max-w-6xl mx-auto px-6 flex justify-between items-center h-14 my-2 rounded-full z-10">
        {/* LOGO */}
        <div className="flex items-center ">
          <h1
            onClick={() => navigate("/")}
            className="text-xl text-[#19334E] font-bold cursor-pointer hover:text-[#FF7F2C] "
          >
            Pls Smile Mr.John Doe{" "}
          </h1>
        </div>
        {/* Mobile Menu Button */}
        <button
          className="md:hidden text-[#19334E] focus:outline-none"
          onClick={() => setIsOpen(!isOpen)}
        >
          {isOpen ? <X size={24} /> : <Menu size={24} />}
        </button>

        {/* Desktop Menu */}
        <div className="hidden md:flex space-x-6 text-[#596269] ">
          {location.pathname === "/" ? (
            <Link to="/" className="hover:text-[#FF7F2C] text-[#FF7F2C]">
              Home
            </Link>
          ) : (
            <Link to="/" className="hover:text-[#FF7F2C]">
              Home
            </Link>
          )}
          {location.pathname === "/mldata" ? (
            <Link to="/mldata" className=" hover:text-[#FF7F2C] text-[#FF7F2C]">
              Machine Learning Dataset
            </Link>
          ) : (
            <Link to="/mldata" className=" hover:text-[#FF7F2C]">
              Machine Learning Dataset
            </Link>
          )}
          {location.pathname === "/nndata" ? (
            <Link to="/nndata" className=" hover:text-[#FF7F2C] text-[#FF7F2C]">
              Neural Network Dataset
            </Link>
          ) : (
            <Link to="/nndata" className=" hover:text-[#FF7F2C]">
              Neural Network Dataset
            </Link>
          )}
        </div>
      </div>

      {/* Mobile Menu */}
      {isOpen && (
        <div
          className={`md:hidden bg-white shadow-[0px_0px_2px_0px_#1a1a1a3d] rounded-lg p-4 flex items-center flex-col w-[85%] space-y-4 text-[#FF7E69] text-center mt-2 mx-4 ${styles.slidein} -z-10 absolute`}
          onClick={()=>setIsOpen(false)}
        >
          {location.pathname === "/" ? (
            <Link to="/" className="hover:text-[#7CE9BF] text-[#7CE9BF]">
              Home
            </Link>
          ) : (
            <Link to="/" className="hover:text-[#7CE9BF]">
              Home
            </Link>
          )}
          {location.pathname === "/mldata" ? (
            <Link to="/mldata" className=" hover:text-[#7CE9BF] text-[#7CE9BF]">
              Machine Learning Dataset
            </Link>
          ) : (
            <Link to="/mldata" className=" hover:text-[#7CE9BF]">
              Machine Learning Dataset
            </Link>
          )}
          {location.pathname === "/nndata" ? (
            <Link to="/nndata" className=" hover:text-[#7CE9BF] text-[#7CE9BF]">
              Neural Network Dataset
            </Link>
          ) : (
            <Link to="/nndata" className=" hover:text-[#7CE9BF]">
              Neural Network Dataset
            </Link>
          )}
        </div>
      )}
    </nav>
  );
};

export default Navbar;
