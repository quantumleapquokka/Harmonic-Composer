import { useEffect, useRef } from "react"
import { OpenSheetMusicDisplay } from "opensheetmusicdisplay"

function SheetViewer({ xml }) {
    const containerRef = useRef(null)  
    const osmdRef = useRef(null)  

    useEffect(() => {
        console.log("SheetViewr XML:", xml)
        if (!xml && !xmlUrl) return  

        if (!osmdRef.current) {
            osmdRef.current = new OpenSheetMusicDisplay(containerRef.current, {
                drawingParameters: "compacttight",
                drawPartNames: true,
                autoResize: true,
            })
        }

        const loadSheet = async () => {
            try {
                await osmdRef.current.load(xml)

                 osmdRef.current.setOptions({
                    drawPartNames: true,
                    drawPartAbbreviations: true,
                })

                // Wait for container to layout
                requestAnimationFrame(async () => {
                await osmdRef.current.render()
                });
            } catch (err) {
                console.error("OSMD error:", err)
                containerRef.current.innerHTML = "<p>Could not load sheet music.</p>"
            }
        }

        loadSheet()  
    }, [xml])  

    return <div ref={containerRef} style={{ width: "100%", maxWidth: "1275px" }}/>  
} 

export default SheetViewer