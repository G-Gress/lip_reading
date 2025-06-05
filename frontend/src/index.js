// frontend/src/index.js
import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./index.css";  // if you have global styles

// Find the <div id="root"></div> in public/index.html
const container = document.getElementById("root");
const root = createRoot(container);
root.render(<App />);
