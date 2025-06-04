#!/bin/bash

# Get the site-packages path for the current Python environment
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")

# Clone the pyaer repository
git clone https://github.com/duguyue100/pyaer.git
cd pyaer

# Build and install the package in development mode (if necessary)
make develop

# Copy the pyaer directory to the site-packages directory
cp -r pyaer "$SITE_PACKAGES"

# Copy the pyaer directory to the site-packages directory
cp -r scripts ../

# Clean up by removing the cloned repository
cd ..
rm -rf pyaer