import { useState, useEffect } from 'react' 
import Header from './Components/Header.jsx' 
import SheetViewer from './Components/SheetViewer.jsx' 
import './App.css' 

function App() {
    // State for all user selections
    const [selectedInstrument, setSelectedInstrument] = useState('piano') 
    const [selectedGenre, setSelectedGenre] = useState('classical') 
    const [selectedEra, setSelectedEra] = useState('2000s') 

    const [chords, setChords] = useState('Cmajor Gmajor Aminor Fmajor')
    const [status, setStatus] = useState('Output will appear here.')
    const [downloadUrl, setDownloadUrl] = useState(null)
    const [isLoading, setIsLoading] = useState(false)
    const [xmlString, setXmlString] = useState(null)
    const [overviewXml, setOverviewXml] = useState(null)

    const handleGenerateMusic = async () => {
        setIsLoading(true)
        setStatus('Composing... This may take a few minutes...')
        setDownloadUrl(null)
        setXmlString(null)

        const chordList = chords.split(' ').filter(c => c)

        if (chordList.length === 0) {
        setStatus('Error: Please enter at least one chord.')
        setIsLoading(false)
        return
        }

        try {
        const response = await fetch('http://127.0.0.1:5000/generate', {
            method: 'POST',
            headers: {
            'Content-Type': 'application/json',
            },
            body: JSON.stringify({
            genre: selectedGenre,
            chords: chordList,
            instrument: selectedInstrument,
            }),
        }) 

        const result = await response.json()

        if (response.ok) {
            setStatus(`${result.message}`)
            setDownloadUrl(`http://127.0.0.1:5000${result.download_url}`)

            const xmlResp = await fetch(`http://127.0.0.1:5000${result.xml_url}`)
            const xmlText = await xmlResp.text()
            setXmlString(xmlText)
        } else {
            setStatus(`Error: ${result.error || 'An unknown error occurred.'}`)
        }
        } catch (error) {
        setStatus('Error: Could not connect to the server. Is it running?')
        console.error('Fetch error:', error)
        }

        setIsLoading(false) 
    } 

    useEffect(() => {
        const fetchLatestOverview = async () => {
            try {
            const resp = await fetch('/output/ai_transformer_composition_piano.musicxml'); // served by frontend
            if (resp.ok) {
                const text = await resp.text();
                setOverviewXml(text);
            }
            } catch (err) {
                console.error('Failed to fetch latest overview:', err);
            }
        };

            fetchLatestOverview();
    }, []);


    return (
        <>
        <Header />
        <main>
            <section id="overview" className="section">
                <h3>Use the AI to generate an original composition based on your specifications.</h3>
                <h1>Last Composition</h1> 
                <div className="sheet-container">
                    {overviewXml ? (
                    <SheetViewer xml={overviewXml} />
                    ) : (
                    <p>No previous composition available.</p>
                    )}
                </div>
            </section>
            
            <h2>Generate your own music below!</h2>

            <section id="generation" className="section generation-layout">
                <div className="output-side">
                    <h2>Output</h2>
                        <div className="sheet-container">
                            {xmlString ? (
                                <SheetViewer xml={xmlString} />
                            ) : (
                                <p>{status}</p>
                            )}

                            {downloadUrl && (
                                <a href={downloadUrl} download className="download-link">
                                Download Your MIDI File
                                </a>
                            )}
                        </div>
                </div>
                
                <div className="settings-side">
                    <h2>Select Settings</h2>
                    <p>Select your desired specifications for what you want to generate.</p>
                    <div className="settings-section">
                        <div className="settings-container">
                            <label>
                                Instrument: <br/>
                                <select value={selectedInstrument} onChange={(e) => setSelectedInstrument(e.target.value)}>
                                <option value="piano">Piano</option>
                                <option value="guitar">Guitar</option>
                                </select>
                            </label>
                            <br/>

                            <label>
                                Genre: <br/>
                                <select value={selectedGenre} onChange={(e) => setSelectedGenre(e.target.value)}>
                                <option value="blues">Blues</option>
                                <option value="jazz">Jazz</option>
                                <option value="classical">Classical</option>
                                <option value="country">Country</option>
                                <option value="pop">Pop</option>
                                </select>
                            </label>
                            <br/>
                            
                            <label>
                                Era: <br/>
                                <select value={selectedEra} onChange={(e) => setSelectedEra(e.target.value)}>
                                <option value="2000">2000s</option>
                                <option value="1700">1700s</option>
                                {/* Add other eras as needed */}
                                </select>
                            </label>
                            <br/>

                            <label>
                                Chord Progression: <br/>
                                <input 
                                type="text" 
                                value={chords} 
                                onChange={(e) => setChords(e.target.value)} 
                                placeholder="e.g. Cmajor Gmajor Aminor"
                                />
                            </label>
                        </div>

                            <div className="iteration-controls">
                                <button>Regenerate Entirely</button>
                                <button>Keep Rhythm, Change Melody</button>
                                <button>Keep Melody, Change Instrumentation</button>
                                <button>Keep Everything, Just Refine</button>
                            </div>

                    </div>
                </div>

                <div className="generate-wrapper">
                    <button className="button" onClick={handleGenerateMusic} disabled={isLoading}>
                    {isLoading ? 'Composing...' : 'Generate Music'}
                    </button>
                </div>
            </section>

            <section id="about" className="section">
            <h2>About / Team</h2>
            <p>CMPM 146 Final Project<br/>AI takes on the role of a collaborative composer, generating, revising, and finalizing music in conjunction with the user.<br/>Brandon Hernandez, Samantha Siew, Shripad Mangavalli, Grishen Hestiyas</p>
            </section>
        </main>
        </>
    )
}

export default App 