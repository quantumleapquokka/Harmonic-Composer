import React, { useState } from 'react'
import Logo from '../assets/logo.svg'
import '../index.css'

function Header() {
    const [menuOpen, setMenuOpen] = useState(false)

    return(
        <header className="site-header">
            {/*Logo and Title*/}
            <div className="logo-container">
                <img src={Logo} alt="Harmonic Composer Logo" className="logo"/>
                <h1 className="site-title">Harmonic Composer</h1>
            </div>

            {/*Side Navigation Menu*/}
            <nav className="navBar">
                <button
                    className={`hamburger ${menuOpen ? "open" : ""}`}
                    onClick={() => setMenuOpen(!menuOpen)}
                    aria-label="Toggle menu"
                >
                    <span />
                    <span />
                    <span />
                </button>

                <ul className={`nav-links ${menuOpen ? "show" : ""}`}>
                    <li><a href="#overview">Overview</a></li>
                    <li><a>Settings</a></li>
                    <li><a href="#about">About/Team</a></li>
                </ul>
            </nav>

        </header>
    )
}

export default Header