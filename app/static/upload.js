const uploadForm = document.querySelector(".upload-form");

uploadForm.addEventListener("submit", async event => {
    event.preventDefault();
    const button = uploadForm.querySelector("button");
    if (button.disabled) return;
    const status = document.getElementById("upload-status");
    button.disabled = true;
    status.hidden = false;
    status.textContent = "Uploading...";
    try {
        const response = await fetch("/upload", {
            method: "POST",
            body: new FormData(uploadForm),
        });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(typeof data.detail === "string" ? data.detail : "Could not upload product.");
        }
        uploadForm.reset();
        status.textContent = `Uploaded "${data.title}". Index it to enable search.`;
        document.dispatchEvent(new Event("product-uploaded"));
    } catch (error) {
        status.textContent = error.message;
    } finally {
        button.disabled = false;
    }
});
