import './App.css'
import { BrowserRouter as Router, Routes, Route } from 'react-router'
import ScrollTop from './assets/component/ScrollTop'
import Navbar from './assets/component/Navbar'
import Home from './assets/page/Home'
import NNdata from './assets/page/NNdata'
import MLdata from './assets/page/MLdata'
function App() {

  return (
    <>
      <Router>
      <ScrollTop/>
        <Navbar/>
        <Routes>
          <Route path="/" element={<><Home /></>}/>
          <Route path="/nndata" element={<><NNdata/></>}/>
          <Route path="/mldata" element={<><MLdata/></>}/>
        </Routes>
      </Router>
    </>
  )
}

export default App
