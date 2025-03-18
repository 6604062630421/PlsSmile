import './App.css'
import { BrowserRouter as Router, Routes, Route } from 'react-router'
import ScrollTop from './assets/component/ScrollTop'
import Home from './assets/page/Home'
import NNdata from './assets/page/nnData'
import MLdata from './assets/page/MLdata'
import Navbar from './assets/component/Navbar'
import Footer from './assets/component/Footer'
import { useCallback, useState } from 'react'
function App() {
  const [poked,setpoke] = useState(false);
  const sendrespone = useCallback((info)=>{
    setpoke(info);
  })
  return (
    <>
      <Router>
      <ScrollTop/>
        <Navbar/>
        <Routes>
          <Route path="/" element={<><Home setPoke={sendrespone} pokeref={poked}/><Footer/></>}/>
          <Route path="/nndata" element={<><NNdata/><Footer/></>}/>
          <Route path="/mldata" element={<><MLdata/><Footer/></>}/>
        </Routes>
      </Router>
    </>
  )
}

export default App
