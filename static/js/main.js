var modelId;
var sessionUUID;
var isSendEnabled = false;

function init()
{
    setSendStatus(false);
    const message = {"type": "ClientJoinedRequest"};
    sendPostRequest(message).then((json) => displayModelSelector(json));
}

function displayModelSelector(models)
{
    //alert("TODO: displayModelSelector" + JSON.stringify(models));

    // TODO: Display model selector, send openModelRequest and poll until it is successful.
    modelId = "OpenAI/GPT 2/SMALL";

    // TODO: It should be triggered by the model selector widget
    const message = {"type": "PollOpenModelRequest",
                    "modelId": modelId};
    sendPostRequest(message).then((json) => startSession(json));
}

function startSession()
{
    const message = {"type": "StartSessionRequest",
                     "modelId": modelId};

    sendPostRequest(message).then((json) => sessionStarted(json));
}

function sessionStarted(json)
{
    sessionUUID = json.sessionUUID;
    setSendStatus(true);
}

function setSendStatus(isEnabled)
{
    isSendEnabled = isEnabled;

    if (isEnabled)
    {
        document.getElementById("sendImage").src = "/open-all-gpt/static/image/send.jpeg";
    }
    else
    {
        //document.getElementById("inputField").value;
        document.getElementById("sendImage").src = "/open-all-gpt/static/image/sendDisabled.jpeg";
    }
}

function checkInput(event)
{
    if (isSendEnabled && event.code === "Enter")
    {
        send();
    }
    return true;
}

function send()
{
    if (isSendEnabled)
    {
        setSendStatus(false);

        // Display input text in the body area
        const text = document.getElementById("inputField").value;

        const html = '<td colspan="2"></td>' +
                   '<td colspan="2">' +
                   '    <div class="inputText">' +
                   '        <p>' + text + '</p>' +
                   '    </div>' +
                   '</td>' +
                   '<td><img src="/open-all-gpt/static/image/user.png" class="messageImage" alt="User"/></td>';

        const bodyTableRef = document.getElementById("content");
        const newRow = bodyTableRef.insertRow();
        newRow.insertAdjacentHTML('beforeend', html);

        // Scroll to the bottom
        const bodyDiv = document.getElementById("bodyDiv");
        bodyDiv.scrollTop = bodyDiv.scrollHeight;

        // Clear the input field
        document.getElementById("inputField").value = "";

        // Send input to server
        const message = {"type": "QueryRequest",
                         "modelId": modelId,
                         "sessionUUID": sessionUUID,
                         "topK": 40,
                         "maxLength": 30,
                         "text": text};

        sendPostRequest(message).then((json) => pollQueryRequest(json));
    }
}

function addOutputText(text)
{
    const html = '<td>' +
               '   <img src="/open-all-gpt/static/image/gpt.png" class="messageImage" alt="GPT"/>' +
               '</td>' +
               '<td colspan="2">' +
               '    <div class="outputText" id="lastOutput">' +
               '        <p>' + text + '</p>' +
               '    </div>' +
               '</td>' +
               '<td colspan="2"></td>';

    const bodyTableRef = document.getElementById("content");
    const newRow = bodyTableRef.insertRow();
    newRow.insertAdjacentHTML('beforeend', html);

    // Scroll to the bottom
    const bodyDiv = document.getElementById("bodyDiv");
    bodyDiv.scrollTop = bodyDiv.scrollHeight;
}

async function sendPostRequest(message)
{
    serverHost = location.host;
    const settings = {
        method: "POST",
        headers: {
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        body: JSON.stringify(message)
    };

    const fetchResponse = await fetch("http://" + serverHost + "/open-all-gpt", settings);
    const data = await fetchResponse.json();
    return data;
}

async function pollPostRequest(message)
{
    serverHost = location.host;
    const settings = {
        method: "POST",
        headers: {
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        body: JSON.stringify(message)
    };

    const fetchResponse = await fetch("http://" + serverHost + "/open-all-gpt", settings);
    const result = await fetchResponse.json();
    return result;
}

const delay = ms => new Promise((resolve) => setTimeout(resolve, ms));

async function pollQueryRequest(json)
{
    const queryUUID = json.queryUUID;

    var attempt = 0;
    while (true)
    {
        // Send PollQueryResult request to server
        var message = {"type": "PollQueryResultRequest",
                         "modelId": modelId,
                         "queryUUID": queryUUID,
                         "attempt": attempt};

        if (attempt == 0)
        {
            addOutputText("...")
        }

        attempt++;

        serverHost = location.host;
            const settings = {
                method: "POST",
                headers: {
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(message)
            };

        const fetchResponse = await fetch("http://" + serverHost + "/open-all-gpt", settings);
        const result = await fetchResponse.json();

        if (result.ready === true)
        {
            finishQueryResult(result);
            return;
        }
        else
        {
            updateQueryResult(result);
            await delay(500);
        }
    }
}

function updateQueryResult(json)
{
    document.getElementById("lastOutput").innerHTML = '<p>' + json.text + '...</p>';
}

function finishQueryResult(json)
{
    document.getElementById("lastOutput").innerHTML = '<p>' + json.text + '</p>';
    document.getElementById("lastOutput").id = "";
    setSendStatus(true);
}