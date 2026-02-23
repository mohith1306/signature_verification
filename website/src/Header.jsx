import React from 'react'
import './Header.css'
const Header = () => {
  return (
    <div>
      <div className='header'>
        Signature Verfication
      </div>
      <div className='hero'>
        <div className = 'hero1'>
            Verify you signature authenticity
        </div>
        <h1>Note: This model is currently under further development and training. The present accuracy stands at 85%, and ongoing improvements are being made to enhance performance.</h1>
        <p>
            upload an image of your signature to determine its authenticity. For best results, use clear and high resolution images.
        </p>
      </div>
    </div>
  )
}

export default Header
