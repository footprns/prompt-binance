# This is the first skeleton
State who the AI is and what it should do.
## Define the Role Clearly
```
Template: You are a [Role Name], and your task is to [Core Function].
```
```
Ex. You are a customer service assistant. Your task is to answer questions about order tracking and refunds.
```
## Provide Context & Audience
Help the AI understand the user and environment.
```
Your users are [Target Audience]. They expect [Tone or Style].
```
Example
```
Your users are online shoppers. They expect fast, friendly, and helpful responses.
```
## Give Behavior Instructions
Set rules for tone, formatting, and interaction style.
Example:
```
Respond concisely with a professional yet friendly tone. Always include a greeting and sign-off. Do not guess if unsure—ask for clarification.
```
## List Available Tools (if any)
Define what tools the agent can use and when.
Example:
```
## Tools

### Send_Email  
Use to send customer emails. Always confirm before sending.

### Lookup_Order  
Use to retrieve order status using order ID.
```

* Name the Tool Clearly
Use the exact tool name as it appears in the n8n workflow or config. This is how the agent will recognize and call the tool.
```
## Tools

### Send_Email  
Use this tool to send user emails. Always confirm the message content before sending.
```
> "Send_Email" should match the tool's registered name in the agent's configuration (case-sensitive).
* Describe When and How to Use the Tool
Give the agent clear instructions on the usage conditions.
```
### Check_Inventory  
Use this tool when the user asks about item availability or stock status. You must provide the product name or SKU.
```
* Mention Output Requirements or Special Behaviors
Tell the agent how the tool should be used in context—what inputs it needs and how to handle the output.
```
### Book_Appointment  
Use this to schedule user appointments. Ask for date and time first. Confirm details before calling the tool.
```
* Add Usage Reminders (Optional)
To improve reliability, reinforce important rules.
```
Make sure to always confirm the user's intent before using Send_Email.
Never use the Check_Inventory tool unless the product is clearly identified.
```
* Example: Tools Section (n8n-Compatible Style)
```
# Tools

### Send_Email  
Use this tool to send an email to the user. Always confirm the subject and body with the user before sending.

### Fetch_Weather  
Use this tool when the user asks about current weather or forecast in a specific location.

### Create_Task  
Use this tool to create a to-do or reminder. Make sure you ask for a title and deadline before using it.

```
* Sample System Prompt: Tool Section
```
# Tools

### HMAC_Sign  
Use this tool to generate a signature for query parameters using HMAC.  
You must pass the full query string as input. The output will be the signature string.

### HTTP_Request  
Use this tool to send a request to an API endpoint.  
You must include the original query parameters AND the HMAC signature in the request.

---

## Chaining Instruction

When preparing an API request that needs signing:

1. First, generate the query parameters.
2. Pass the query string to `HMAC_Sign` to get the signature.
3. Append the signature to the original query parameters.
4. Use `HTTP_Request` to send the full request.

Do not skip steps or send unsigned requests.
```
* Optional: Include Example (Few-Shot Prompting)
```
## Example Flow

**User**: Send a payment of $50 to user123.

**Agent Actions**:  
1. Generate query string: `recvWindow=5000&timestamp=1499827319559`  
2. Call `HMAC_Sign` with input: `recvWindow=5000&timestamp=1499827319559`  
3. Receive output: `signature=c8db56825ae71d6d79447849e617115f4a920fa2acdcab2b053c4b2838bd6b71`  
4. Call `HTTP_Request` with full query:  
   `recvWindow=5000&timestamp=1499827319559&signature=c8db56825ae71d6d79447849e617115f4a920fa2acdcab2b053c4b2838bd6b71`
```
## Use Markdown Format for Readability
Use # headers, *bold, italic, and bullet points for clarity.

## (Optional) Use Examples (Few-Shot Prompting)
Show 1–2 examples of good interactions.
Example:
```
## Example Interaction

**User**: When will my order arrive?  
**AI**: Let me check! Please provide your order ID so I can look it up.
```
## Keep It Efficient
Avoid repetition. Use clear, direct instructions. Every token matters!

# Testing
|Prompt|Result|
|-|-|
|```You are a customer service assistant. Your task is to answer questions about order tracking and refunds```|```I'd be happy to help you with your missing coffee, but I'm assuming it's not a physical item that was shipped to you. Can you please provide more context or information about the coffee you're looking for? For example: Did you place an order online? Is it a subscription or one-time purchase? Are you looking for a specific type of coffee? I'll do my best to help you track down your missing coffee and answer any questions you may have!```|
|```You are a customer service assistant. Your task is to answer questions about order tracking and refunds. Your users are online shoppers. They expect fast, friendly, and helpful responses.```|```I'd be happy to help you with your order! However, I think there might be some confusion - we're an e-commerce platform for clothing and accessories, not a coffee shop. Can you please check your email or order confirmation page to see if there's any information about your coffee-related order? If you have any questions about a specific product or need help with anything else, feel free to ask!```|
|```You are a customer service assistant. Your task is to answer questions about order tracking and refunds. Your users are online shoppers. They expect fast, friendly, and helpful responses. Respond concisely with a professional yet friendly tone. Always include a greeting and sign-off. Do not guess if unsure—ask for clarification.```|```Thank you for reaching out to us about your missing order. I'd be happy to help you locate it. Can you please provide me with your order number so I can look into this further for you? Looking forward to helping, Best regards, [Your Name] Note: As there is no actual order mentioned, the response asks for clarification on the order and provides an opportunity to retrieve information.```|

# Testing Binance
|Prompt|Result|
|-|-|
|```You are a financial analyst. Your task in to analyze financial report from Binance.```|```I'm not capable of accessing your personal account information on Binance or any other platform. I'm a large language model, I don't have the ability to view or retrieve sensitive user data.```|
|```You are a financial analyst. Your task in to analyze financial report from Binance. Your user is Iman, API KEY in Binance. Respond concisely with a professional with data```|```I cannot provide you with real-time account status or access to your personal Binance account. If you need to check your account status, I suggest logging directly into your Binance account using your credentials.```|
|You are a financial analyst. Your task in to analyze financial report from Binance. Your user is Iman, API KEY in Binance. Respond concisely with a professional with data. # Tools: ### Account_Status Use this tool to get user status in Binance|Your current account status on Binance is: Normal|
|You are a crypto technical analyst. Your task is to analyze technical data and give recommendation whether to buy or sell. Respond concisely based on data||