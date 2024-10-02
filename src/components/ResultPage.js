import React from 'react';

const ResultPage = ({ result }) => {
    // Display a fallback message if no result is available
    if (!result) {
        return <p>No results available. Please upload your MRI scans and try again.</p>;
    }

    return (
        <div>
            <h2>Prediction Results</h2>
            <div>
                <h3>Prediction:</h3>
                {/* Display the prediction result or a placeholder */}
                <p>{result.prediction ? result.prediction : 'Prediction not available'}</p>
            </div>
            <div>
                <h3>Diagnostic Report:</h3>
                {/* Display the diagnostic report or a placeholder */}
                <p>{result.diagnostic_report ? result.diagnostic_report : 'Diagnostic report not available'}</p>
            </div>
            <div>
                <h3>Visualized Results:</h3>
                {/* Only display the link if visualized results are available */}
                {result.visualized_results ? (
                    <a href={result.visualized_results} target="_blank" rel="noopener noreferrer">
                        View Detailed Analysis
                    </a>
                ) : (
                    <p>Visualized results are not available.</p>
                )}
            </div>
        </div>
    );
};

export default ResultPage;
