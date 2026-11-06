const collectionList = document.getElementById("collection-list");

function updateSelection() {
    document.querySelectorAll("[data-collection-id]").forEach(link => {
        link.setAttribute("aria-current", link.dataset.collectionId === (new URL(window.location.href).searchParams.get("collection_id") || "") ? "page" : "false");
    });
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
        document.dispatchEvent(new CustomEvent("collections-loaded", {detail: collections}));
        updateSelection();
    } catch (error) {
        collectionList.textContent = "Could not load collections. Refresh the page to try again.";
    }
}

document.addEventListener("gallery-updated", updateSelection);
document.addEventListener("products-changed", loadCollections);
document.addEventListener("agent-response", loadCollections);
loadCollections();
