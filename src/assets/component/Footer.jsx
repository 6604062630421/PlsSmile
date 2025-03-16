const Footer = () => {
    return (
      <footer className="bg-[#1F2F4D] text-white text-sm px-10 flex flex-col md:flex-row justify-between items-center py-6">
        <div className="flex items-center space-x-4 ">
          <div>
            <p>Pummipat Pakdeeto © 2025 All right Reserves</p>
            <p>For practice training Machine Learning and Neural Network</p>
          </div>
        </div>
        <div className="text-right mt-4 md:mt-0">
          <p>Contract issues: <a href="mailto:Support@northclub.kmutnb.ac.th" className="underline">s6604062630421@email.kmutnb.ac.th</a></p>
          <p><br/></p>
        </div>
      </footer>
    );
  };
  
  export default Footer;