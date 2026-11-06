const indexForm = document.getElementById("index-form");

indexForm.addEventListener("submit", async event => {
    event.preventDefault();
    const button = indexForm.querySelector("button");
    if (button.disabled) return;
    const status = document.getElementById("index-status");
    button.disabled = true;
    status.hidden = false;
    status.textContent = "Indexing...";
    try {
        const response = await fetch("/run-pipeline", {method: "POST"});
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
            throw new Error(typeof data.detail === "string" ? data.detail : "Could not index products. Please try again.");
        }
        status.textContent = `Indexed ${data.indexed} products.`;
        document.dispatchEvent(new Event("products-indexed"));
    } catch (error) {
        status.textContent = error.message;
    } finally {
        button.disabled = false;
    }
});
