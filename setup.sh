#!/bin/bash

# Add all files
git add .

# Commit with message
git commit -m "Add complete project files"

# Force push to main branch
git push -f origin main

echo "Files pushed to GitHub successfully!" 