import React from 'react'
import Logo from '../assets/logo.svg'
import '../index.css'

function Header() {
    return(
        <div id="appRoot" >
            <header className="site-header">
                <div className="logo-container">
                    <img src={Logo} alt="Harmonic Composer Logo" className="logo"/>
                    <h1 className="site-title">Harmonic Composer</h1>
                </div>
            </header>
        </div>
        
    )
}

export default Header