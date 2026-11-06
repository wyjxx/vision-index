let galleryUrl = new URL(window.location.href);
let galleryRequest = 0;

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
        document.dispatchEvent(new Event("gallery-updated"));
        status.textContent = "";
    } catch (error) {
        if (requestId === galleryRequest) status.textContent = "Could not load products. Please try again.";
    }
}

document.querySelector(".search-form").addEventListener("submit", event => {
    event.preventDefault();
    const query = new FormData(event.currentTarget).get("q").trim();
    if (!query) return;
    const url = new URL("/search/clip", window.location.origin);
    url.searchParams.set("q", query);
    showGallery(url);
});

document.addEventListener("product-uploaded", () => {
    showGallery(new URL("/", window.location.origin));
});

document.addEventListener("products-indexed", () => {
    showGallery(galleryUrl);
});

document.addEventListener("click", event => {
    const link = event.target.closest(".collection-folder, #all-products, #product-gallery .pagination a, #product-gallery .section-heading a");
    if (!link || event.ctrlKey || event.metaKey || event.shiftKey || event.button !== 0) return;
    event.preventDefault();
    showGallery(link.href);
});

document.addEventListener("collections-loaded", event => {
    const selectedId = galleryUrl.searchParams.get("collection_id");
    if (!selectedId) return;
    const exists = event.detail.some(collection => String(collection.id) === selectedId);
    showGallery(exists ? galleryUrl : new URL("/", window.location.origin));
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
        document.dispatchEvent(new Event("products-changed"));
    } catch (error) {
        document.getElementById("collection-status").textContent = error.message;
    } finally {
        remove.disabled = false;
    }
});
