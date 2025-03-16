import { useState } from "react";
import CopyToClipboard from "react-copy-to-clipboard";
import { Copy } from "lucide-react";
import { Prism as Syntaxhighlight} from "react-syntax-highlighter";
import { github } from "react-syntax-highlighter/dist/esm/styles/hljs";
const Copybox = ({text,lang}) =>{
    const [hov1,setHover1] = useState(false);
    return(
        <div className="cursor-pointer pt-3 text-[13px]">
        <CopyToClipboard text={text}>
            <div
              onMouseEnter={() => setHover1(true)}
              onMouseLeave={() => setHover1(false)}
              className="relative px-5 py-2 rounded-[7px] border-1 border-[#1a1a1a1d] fit"
            >
                <Syntaxhighlight language='python' style={github}customStyle={{ backgroundColor: '#fff' }}>
                    {text}
                </Syntaxhighlight>
              <div
                className={`absolute right-0 top-0 h-full px-3 w-[40px] ${
                  hov1 ? "opacity-100" : "opacity-0"
                } ease-out transition-opacity duration-200`}
              >
                <div className="h-full flex justify-center items-center ">
                  <Copy className="opacity-50 hover:opacity-100"/>
                </div>
              </div>
            </div>
          </CopyToClipboard>
        </div>
    )
}


export default Copybox;