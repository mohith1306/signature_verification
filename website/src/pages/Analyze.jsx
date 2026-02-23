// import React, { useState } from "react";
// import axios from "axios";
// import "../App.css";

// const Analyze = () => {
//   const [file1, setFile1] = useState(null);
//   const [file2, setFile2] = useState(null);
//   const [result, setResult] = useState(null);
//   const [loading, setLoading] = useState(false);

//   const handleSubmit = async (e) => {
//     e.preventDefault();
//     if (!file1 || !file2) {
//       alert("Please upload both signature images");
//       return;
//     }

//     const formData = new FormData();
//     formData.append("image1", file1);
//     formData.append("image2", file2);

//     try {
//       setLoading(true);
//       const res = await axios.post("http://localhost:5000/predict", formData, {
//         headers: { "Content-Type": "multipart/form-data" },
//       });
//       console.log("API Response:", res.data);
//       setResult(res.data.similarity_score);
//     } catch (err) {
//       console.error("Error during API call:", err);
//       alert("Error analyzing signatures");
//     } finally {
//       setLoading(false);
//     }
//   };

//   return (
//     <div className="analyze-page">
//       <h2>Analyze Signatures</h2>
//       <form onSubmit={handleSubmit}>
//         <label>Signature 1:</label>
//         <input
//           type="file"
//           accept="image/*"
//           onChange={(e) => setFile1(e.target.files[0])}
//         />
//         <label>Signature 2:</label>
//         <input
//           type="file"
//           accept="image/*"
//           onChange={(e) => setFile2(e.target.files[0])}
//         />
//         <button type="submit" disabled={loading}>
//           {loading ? "Analyzing..." : "Submit"}
//         </button>
//       </form>

//       {result !== null && (
//         <div className="result">
//           <h3>Match Percentage: {result.toFixed(2)}%</h3>
//         </div>
//       )}
//     </div>
//   );
// };

// export default Analyze;

import React from "react";
import { useLocation, useNavigate } from "react-router-dom";
// import "../App.css";

const Analyze = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const similarity = location.state?.similarity;

  if (similarity === undefined) {
    return (
      <div className="analyze-page">
        <h3>No analysis result found.</h3>
        <p>Please upload two signatures to analyze.</p>
        <button onClick={() => navigate("/")}>Go Back</button>
      </div>
    );
  }

  return (
    <div className="analyze-page">
      <h2>Signature Match Result</h2>
      <h3 style={{ marginTop: "20px" }}>
        ✅ Match Percentage: {similarity.toFixed(2)}%
      </h3>
      <button
        onClick={() => navigate("/")}
        style={{
          marginTop: "30px",
          padding: "10px 20px",
          background: "#007bff",
          color: "white",
          border: "none",
          borderRadius: "5px",
          cursor: "pointer",
        }}
      >
        Try Again
      </button>
    </div>
  );
};

export default Analyze;
