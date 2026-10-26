const form = document.getElementById("chat-form");
const input = document.getElementById("chat-input");
const button = form.querySelector("button");
const messages = document.getElementById("chat-messages");
const status = document.getElementById("chat-status");
let threadId = null;

function addMessage(role, text) {
    const message = document.createElement("p");
    message.className = `chat-message ${role.toLowerCase()}`;
    message.textContent = `${role}: ${text}`;
    messages.append(message);
    messages.scrollTop = messages.scrollHeight;
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const reply = input.value.trim();
    if (!reply || input.disabled) return;
    input.disabled = button.disabled = true;
    status.textContent = "Thinking...";
    addMessage("You", reply);
    try {
        const url = threadId ? `/agent/${threadId}/resume` : "/agent/start";
        const response = await fetch(url, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(threadId ? {reply} : {question: reply}),
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
            if (response.status === 404 || response.status === 409) threadId = null;
            throw new Error(typeof data.detail === "string" ? data.detail : `Request failed (${response.status}).`);
        }
        threadId = data.thread_id;
        document.dispatchEvent(new Event("agent-response"));
        if (data.type === "write_approval") {
            const calls = data.tool_calls.map(call => `${call.name}\n${JSON.stringify(call.args, null, 2)}`).join("\n\n");
            addMessage("Assistant", `${calls}\n\nEnter y to confirm, or other text to cancel.`);
        } else if (data.type === "end") {
            threadId = null;
            addMessage("Assistant", "Conversation ended. Send a message to start a new one.");
        } else {
            addMessage("Assistant", data.answer);
        }
        input.value = "";
    } catch (error) {
        addMessage("Error", error.message);
    } finally {
        input.disabled = button.disabled = false;
        status.textContent = "";
        input.focus();
    }
});
