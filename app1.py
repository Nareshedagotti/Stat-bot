import telebot
import google.generativeai as genai
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Bot configuration
bot = telebot.TeleBot("7928366370:AAGsDNu5nwaHb1wJhv8eOqCboTOE9-xE17g", parse_mode='HTML')

# Configure Gemini API
genai.configure(api_key='AIzaSyDW82qFu0rK1niSYJHULmz1guiFf6mIfPk')
model = genai.GenerativeModel('gemini-1.5-flash')

# Define allowed topics for StatFusionAI
ALLOWED_TOPICS = [
    'statistics', 'stats', 'probability', 'machine learning', 'ml', 'artificial intelligence', 
    'ai', 'deep learning', 'dl', 'neural network', 'data science', 'data analysis', 
    'mathematics', 'math', 'calculus', 'linear algebra', 'regression', 'classification',
    'clustering', 'algorithm', 'python', 'r programming', 'pandas', 'numpy', 'scikit-learn',
    'tensorflow', 'pytorch', 'keras', 'hypothesis testing', 'anova', 'correlation',
    'data visualization', 'feature engineering', 'model evaluation', 'cross validation',
    'overfitting', 'underfitting', 'bias', 'variance', 'optimization', 'gradient descent',
    'supervised learning', 'unsupervised learning', 'reinforcement learning', 'nlp',
    'computer vision', 'time series', 'forecasting', 'bayesian', 'frequentist',
    'distribution', 'sampling', 'confidence interval', 'p-value', 'significance',
    'dataset', 'feature', 'target', 'training', 'testing', 'validation'
]

def is_relevant_question(text):
    """Check if the question is related to allowed topics"""
    text_lower = text.lower()
    
    # Check for mathematical symbols/expressions
    math_patterns = [r'\d+[\+\-\*/]\d+', r'[xy]\s*=', r'\b(sum|mean|median|mode|std|var)\b',
                    r'\b(matrix|vector|derivative|integral)\b', r'[∑∫∂∇αβγδλμσπ]']
    
    for pattern in math_patterns:
        if re.search(pattern, text_lower):
            return True
    
    # Check for allowed topic keywords
    return any(topic in text_lower for topic in ALLOWED_TOPICS)

def generate_system_prompt():
    """Generate system prompt for StatFusionAI"""
    return """You are StatFusionAI, a specialized AI assistant for data science, statistics, mathematics, machine learning, artificial intelligence, and deep learning questions.

CRITICAL GUIDELINES:
1. Keep responses under 200 words maximum
2. Be concise and direct
3. Use markdown formatting (bold, italics, code blocks)
4. Focus on key points only
5. Include brief code examples if needed
6. Start with a friendly greeting
7. End with "Happy to help further! 🤖"

Your expertise includes:
- Statistics & Probability
- Machine Learning & AI
- Deep Learning & Neural Networks
- Data Science & Programming
- Mathematics

Format: Use **bold**, *italics*, and `code` formatting appropriately."""

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    welcome_text = """
🤖 <b>Hello! Welcome to StatFusionAI Bot!</b> 👋📊

<b>I'm your friendly assistant for:</b>
📈 Statistics & Probability
🤖 Machine Learning & AI
🧠 Deep Learning
📊 Data Science & Analytics
🔢 Mathematics
💻 Programming (Python/R)

<b>Quick Commands:</b>
/start - This welcome message
/topics - See all supported topics

<b>Just ask me anything about:</b>
Data science, ML algorithms, statistics, math problems, Python/R code, or AI concepts!

<i>Note: I focus only on data science related topics! 😊</i>

Ready to help you! 🚀
"""
    bot.reply_to(message, welcome_text)

@bot.message_handler(commands=['topics'])
def show_topics(message):
    topics_text = """
📚 <b>Hey! Here are my expertise areas:</b> 👋

🔸 <b>Statistics & Probability</b>
   • Hypothesis Testing, ANOVA, Distributions
   • Confidence Intervals, P-values
   • Bayesian & Frequentist Statistics

🔸 <b>Machine Learning</b>
   • Regression, Classification, Clustering
   • Model Evaluation, Cross-validation
   • Overfitting/Underfitting Solutions

🔸 <b>Deep Learning & AI</b>
   • Neural Networks, CNNs, RNNs
   • TensorFlow, PyTorch, Keras
   • NLP, Computer Vision

🔸 <b>Data Science</b>
   • Data Preprocessing & EDA
   • Feature Engineering
   • Data Visualization

🔸 <b>Mathematics</b>
   • Linear Algebra, Calculus
   • Optimization, Matrix Operations

🔸 <b>Programming</b>
   • Python (Pandas, NumPy, Scikit-learn)
   • R Programming, SQL

Ask me anything from these topics! 😊🚀
"""
    bot.reply_to(message, topics_text)

def check_gratitude(text):
    """Check if user is saying thank you"""
    gratitude_words = ['thank', 'thanks', 'thx', 'appreciate', 'grateful', 'awesome', 'great', 'perfect', 'excellent']
    return any(word in text.lower() for word in gratitude_words)

def convert_markdown_to_html(text):
    """Convert markdown formatting to HTML for Telegram"""
    # Bold: **text** -> <b>text</b>
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    # Italic: *text* -> <i>text</i>
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    # Code: `text` -> <code>text</code>
    text = re.sub(r'`(.*?)`', r'<code>\1</code>', text)
    # Code blocks: ```text``` -> <pre>text</pre>
    text = re.sub(r'```(.*?)```', r'<pre>\1</pre>', text, flags=re.DOTALL)
    return text

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    try:
        user_question = message.text.strip()
        
        # Check for gratitude first
        if check_gratitude(user_question):
            gratitude_responses = [
                "🤗 You're welcome! I'm here to help with your data science questions anytime! 🚀",
                "😊 Glad I could help! Feel free to ask more stats/ML questions! 📊",
                "🙌 Happy to assist! I'm always here for your AI/ML queries! 🤖",
                "✨ You're welcome! Ready to tackle more data science challenges! 💪",
                "🎉 Pleasure helping you! Ask me anything about stats, ML, or AI! 🧠"
            ]
            import random
            bot.reply_to(message, random.choice(gratitude_responses))
            return
        
        # Check if question is relevant to StatFusionAI
        if not is_relevant_question(user_question):
            off_topic_response = """
🚫 <b>Hi there!</b> I'm StatFusionAI, specialized only in:

📊 Data Science & Statistics
🤖 Machine Learning & AI  
🧠 Deep Learning
🔢 Mathematics
💻 Data Science Programming

Please ask questions related to these topics only! 😊

Use /topics to see all supported areas.
"""
            bot.reply_to(message, off_topic_response)
            return
        
        # Send typing indicator
        bot.send_chat_action(message.chat.id, 'typing')
        
        # Prepare the prompt for Gemini
        system_prompt = generate_system_prompt()
        full_prompt = f"{system_prompt}\n\nUser Question: {user_question}"
        
        # Generate response using Gemini
        response = model.generate_content(full_prompt)
        
        if response and response.text:
            # Convert markdown to HTML
            html_response = convert_markdown_to_html(response.text)
            
            # Check if response is too long (keep under 200 words approximately)
            if len(html_response) > 1500:  # Approximate 200 words
                html_response = html_response[:1500] + "... \n\n<i>Ask for more details if needed!</i> 🤖"
            
            bot.reply_to(message, html_response)
        else:
            bot.reply_to(message, "❌ Hey! I couldn't generate a response. Please try rephrasing your question! 😊")
            
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        error_response = """
❌ <b>Oops! Something went wrong!</b> 😅

Please try again with a simpler question.
I'm here to help with your data science queries! 🤖
"""
        bot.reply_to(message, error_response)

# Error handler
@bot.message_handler(content_types=['photo', 'document', 'audio', 'video', 'voice', 'sticker'])
def handle_media(message):
    media_response = """
📎 <b>Hi there!</b> 👋

I can only process text questions about:
📊 Data Science | ML | AI | Stats | Math

<b>For data files:</b>
• Describe your dataset in text
• Ask specific analysis questions
• Request code examples

<b>For math images:</b>
• Type out the problem in text

Ready to help with your questions! 😊🤖
"""
    bot.reply_to(message, media_response)

if __name__ == "__main__":
    print("🚀 StatFusionAI Bot is starting...")
    print("📊 Specialized in: Data Science | ML | AI | Stats | Math")
    print("🤖 Bot is running and ready to help!")
    
    # Start polling
    try:
        bot.infinity_polling(timeout=60, long_polling_timeout=60)
    except Exception as e:
        logger.error(f"Bot polling error: {e}")
        print("❌ Bot stopped due to an error. Check logs for details.")