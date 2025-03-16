import './App.css'
import { BrowserRouter as Router, Routes, Route } from 'react-router'
import ScrollTop from './assets/component/ScrollTop'
import Home from './assets/page/Home'
import NNdata from './assets/page/NNdata'
import MLdata from './assets/page/MLdata'
import Navbar from './assets/component/Navbar'
import Footer from './assets/component/Footer'
function App() {

  return (
    <>
      <Router>
      <ScrollTop/>
        <Navbar/>
        <Routes>
          <Route path="/" element={<><Home /><Footer/></>}/>
          <Route path="/nndata" element={<><NNdata/><Footer/></>}/>
          <Route path="/mldata" element={<><MLdata/><Footer/></>}/>
        </Routes>
      </Router>
    </>
  )
}

export default App
