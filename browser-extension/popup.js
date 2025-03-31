document.addEventListener("DOMContentLoaded", function () {
    let classifyBtn = document.getElementById("classify-btn");
    let selectedTextArea = document.getElementById("selected-text");
    let mainResultParagraph = document.getElementById("main-result");
    let multiLabelSection = document.getElementById("multi-label-section");
    let multiLabelResultParagraph = document.getElementById("multi-label-result");
    let probabilitiesList = document.getElementById("probabilities-list");

    let userTyped = false; // Flag to track if the user typed manually

    // Function to get selected text on the page
    function getSelectedText() {
        return window.getSelection().toString();
    }

    // Detect user input in the textarea
    selectedTextArea.addEventListener("input", function () {
        userTyped = true; // Mark as manually edited
    });

    // Get the selected text from the active tab, only if the user hasn't typed anything
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        if (tabs.length === 0) return;

        chrome.scripting.executeScript(
            {
                target: { tabId: tabs[0].id },
                function: getSelectedText
            },
            (results) => {
                if (!userTyped && results && results.length > 0 && results[0].result) {
                    selectedTextArea.value = results[0].result.trim(); // Trim to remove extra spaces
                    selectedTextArea.dataset.ready = "true"; // Mark as ready
                }
            }
        );
    });

    classifyBtn.addEventListener("click", function () {
        let selectedText = selectedTextArea.value.trim();

        // Ensure text has actually been entered
        if (!selectedText) {
            mainResultParagraph.textContent = "Please enter or select text.";
            return;
        }

        // Send the text to the Flask server for classification
        fetch("http://127.0.0.1:5000/classify", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: selectedText })
        })
        .then(response => response.json())
        .then(data => {
            // Display main classification
            mainResultParagraph.textContent = `Classification: ${data.class}`;

            // Handle multi-label classification
            if (data.multilabel) {
                multiLabelSection.style.display = "block"; // Show the section
                multiLabelResultParagraph.textContent = `Sub-category: ${data.multilabel.label}`;
                
                // Populate probability list
                probabilitiesList.innerHTML = "";
                for (let category in data.multilabel.probabilities) {
                    let listItem = document.createElement("li");
                    listItem.textContent = `${category}: ${(data.multilabel.probabilities[category] * 100).toFixed(2)}%`;
                    probabilitiesList.appendChild(listItem);
                }
            } else {
                multiLabelSection.style.display = "none"; // Hide if no multi-label result
            }
        })
        .catch(error => {
            console.error("Error:", error);
            mainResultParagraph.textContent = "Error processing classification.";
        });
    });
});
