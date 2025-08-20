// src/App.jsx

import { useState } from 'react';
import Header from './Components/Header.jsx';
import './App.css';

function App() {
  // State for all user selections
  const [selectedInstrument, setSelectedInstrument] = useState('piano');
  const [selectedGenre, setSelectedGenre] = useState('classical');
  const [selectedEra, setSelectedEra] = useState('2000s');

  // NEW: State for the chord progression and UI feedback
  const [chords, setChords] = useState('Cmajor Gmajor Aminor Fmajor');
  const [status, setStatus] = useState('Output will appear here.');
  const [downloadUrl, setDownloadUrl] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  // Replace your existing function with this corrected version
  const handleGenerateMusic = async () => {
    setIsLoading(true);
    setStatus('Composing... This may take a few minutes...');
    setDownloadUrl(null);

    const chordList = chords.split(' ').filter(c => c);

    if (chordList.length === 0) {
      setStatus('Error: Please enter at least one chord.');
      setIsLoading(false);
      return;
    }

    try {
      // FIX 1: Ensure this URL is exactly '127.0.0.1'
      const response = await fetch('http://127.0.0.1:5000/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          genre: selectedGenre,
          chords: chordList,
        }),
      });

      const result = await response.json();

      if (response.ok) {
        setStatus(`✅ ${result.message}`);
        // FIX 2: Ensure the download link is also built with the correct URL
        setDownloadUrl(`http://127.0.0.1:5000${result.download_url}`);
      } else {
        setStatus(`Error: ${result.error || 'An unknown error occurred.'}`);
      }
    } catch (error) {
      setStatus('Error: Could not connect to the server. Is it running?');
      console.error('Fetch error:', error);
    }

    setIsLoading(false);
  };

  return (
    <>
      <Header />
      <main>
        <section id="overview" className="section">
            <h3>Use the AI to generate an original composition based on your specifications.</h3>
            <h1>Last Composition</h1> 
        </section>

        <section id="generation" className="section generation-layout">
            <div className="output-side">
              <h2>Output</h2>
              <div className="composition-output">
                  {/* NEW: Status and download link will appear here */}
                  <div className="status-container">
                    <p>{status}</p>
                    {downloadUrl && (
                      <a href={downloadUrl} download className="download-link">
                        Download Your MIDI File
                      </a>
                    )}
                  </div>
                  <div className="iteration-controls">
                      <button>Regenerate Entirely</button>
                      <button>Keep Rhythm, Change Melody</button>
                      <button>Keep Melody, Change Instrumentation</button>
                      <button>Keep Everything, Just Refine</button>
                  </div>
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
                            {/* FIX: Corrected value and onChange for Genre */}
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
                            {/* FIX: Corrected value and onChange for Era */}
                            <select value={selectedEra} onChange={(e) => setSelectedEra(e.target.value)}>
                              <option value="2000s">2000s</option>
                              {/* Add other eras as needed */}
                            </select>
                        </label>
                        <br/>

                        {/* NEW: Added input for chord progression */}
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
                </div>
            </div>

            <div className="generate-wrapper">
                {/* FIX: Connected the button to the API call function and disabled it while loading */}
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

export default App;