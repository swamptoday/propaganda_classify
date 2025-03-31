chrome.contextMenus.create({
    id: "classifyText",
    title: "Classify Selected Text",
    contexts: ["selection"]
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
    if (info.menuItemId === "classifyText") {
        let selectedText = info.selectionText;

        fetch("http://127.0.0.1:5000/classify", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text: selectedText })
        })
        .then(response => response.json())
        .then(data => {
            let message = `Classification: ${data.class}`;

            // If multi-label classification is available, add it to the message
            if (data.multilabel) {
                message += `\nSub-category: ${data.multilabel.label}`;
                message += `\nProbabilities:\n`;
                for (let category in data.multilabel.probabilities) {
                    message += `${category}: ${data.multilabel.probabilities[category].toFixed(3)}\n`;
                }
            }

            alert(message); // Display result as an alert (can be modified to use a popup)
        })
        .catch(error => {
            console.error("Error:", error);
            alert("Error processing classification.");
        });
    }
});
