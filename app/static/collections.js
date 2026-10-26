const collectionList = document.getElementById("collection-list");
let galleryUrl = new URL(window.location.href);
let galleryRequest = 0;

function updateSelection() {
    document.querySelectorAll("[data-collection-id]").forEach(link => {
        link.setAttribute("aria-current", link.dataset.collectionId === (galleryUrl.searchParams.get("collection_id") || "") ? "page" : "false");
    });
}

async function showGallery(url) {
    const requestId = ++galleryRequest;
    const status = document.getElementById("collection-status");
    status.textContent = "Loading products...";
    try {
        const response = await fetch(url);
        if (!response.ok) throw new Error("Could not load products.");
        const page = new DOMParser().parseFromString(await response.text(), "text/html");
        if (requestId !== galleryRequest) return;
        document.getElementById("product-gallery").replaceWith(page.getElementById("product-gallery"));
        galleryUrl = new URL(url, window.location.origin);
        window.history.replaceState(null, "", galleryUrl);
        updateSelection();
        status.textContent = "";
    } catch (error) {
        if (requestId === galleryRequest) status.textContent = "Could not load products. Please try again.";
    }
}

async function loadCollections() {
    try {
        const response = await fetch("/collections");
        if (!response.ok) throw new Error("Could not load collections.");
        const collections = await response.json();
        const folders = collections.map(collection => {
            const folder = document.createElement("a");
            folder.className = "collection-folder";
            folder.href = `/?collection_id=${collection.id}`;
            folder.dataset.collectionId = String(collection.id);
            folder.setAttribute("role", "listitem");
            const icon = document.createElement("span");
            icon.className = "folder-icon";
            icon.setAttribute("aria-hidden", "true");
            icon.textContent = "📁";
            const copy = document.createElement("div");
            copy.className = "folder-copy";
            const name = document.createElement("strong");
            name.textContent = collection.name;
            name.title = collection.name;
            const count = document.createElement("span");
            count.textContent = `${collection.product_count} products`;
            copy.append(name, count);
            folder.append(icon, copy);
            return folder;
        });
        collectionList.replaceChildren(...folders);
        if (!folders.length) collectionList.textContent = "No collections yet.";
        updateSelection();
    } catch (error) {
        collectionList.textContent = "Could not load collections. Refresh the page to try again.";
    }
}

document.querySelector(".search-form").addEventListener("submit", event => {
    event.preventDefault();
    const query = new FormData(event.currentTarget).get("q").trim();
    if (!query) return;
    const url = new URL(event.currentTarget.action);
    url.searchParams.set("q", query);
    showGallery(url);
});

document.addEventListener("click", event => {
    const link = event.target.closest(".collection-folder, #all-products, #product-gallery .pagination a, #product-gallery .section-heading a");
    if (!link || event.ctrlKey || event.metaKey || event.shiftKey || event.button !== 0) return;
    event.preventDefault();
    showGallery(link.href);
});
document.addEventListener("agent-response", () => {
    loadCollections();
    if (galleryUrl.searchParams.has("collection_id")) showGallery(galleryUrl);
});

document.addEventListener("click", async event => {
    const remove = event.target.closest("[data-delete-product]");
    if (!remove || remove.disabled || !window.confirm(`Delete "${remove.dataset.productTitle}" and its image?`)) return;
    remove.disabled = true;
    try {
        const response = await fetch(`/products/${encodeURIComponent(remove.dataset.deleteProduct)}`, {method: "DELETE"});
        if (!response.ok && response.status !== 404) throw new Error("Could not delete product. If indexing is running, try again later.");
        const url = new URL(galleryUrl);
        url.searchParams.set("page", "1");
        await showGallery(url);
        loadCollections();
    } catch (error) {
        document.getElementById("collection-status").textContent = error.message;
    } finally {
        remove.disabled = false;
    }
});
loadCollections();
