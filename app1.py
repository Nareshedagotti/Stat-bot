import telebot
import google.generativeai as genai
import re
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get tokens from environment variables (safer for deployment)
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

if not TELEGRAM_BOT_TOKEN or not GEMINI_API_KEY:
    print("❌ Please set TELEGRAM_BOT_TOKEN and GEMINI_API_KEY environment variables")
    exit(1)

# ===== INITIALIZE COMPONENTS =====
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN, parse_mode='Markdown')
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

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
3. Use markdown formatting (*bold*, _italics_, `code blocks`)
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

Format: Use *bold*, _italics_, and `code` formatting appropriately for Telegram markdown."""

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    welcome_text = """
🤖 *Hello! Welcome to StatFusionAI Bot!* 👋📊

*I'm your friendly assistant for:*
📈 Statistics & Probability
🤖 Machine Learning & AI
🧠 Deep Learning
📊 Data Science & Analytics
🔢 Mathematics
💻 Programming (Python/R)

*Quick Commands:*
/start - This welcome message
/topics - See all supported topics

*Just ask me anything about:*
Data science, ML algorithms, statistics, math problems, Python/R code, or AI concepts!

_Note: I focus only on data science related topics!_ 😊

Ready to help you! 🚀
"""
    bot.reply_to(message, welcome_text)

@bot.message_handler(commands=['topics'])
def show_topics(message):
    topics_text = """
📚 Hey! Here are my expertise areas: 👋

🔸 *Statistics & Probability*
   • Hypothesis Testing, ANOVA, Distributions
   • Confidence Intervals, P-values
   • Bayesian & Frequentist Statistics

🔸 *Machine Learning*
   • Regression, Classification, Clustering
   • Model Evaluation, Cross-validation
   • Overfitting/Underfitting Solutions

🔸 *Deep Learning & AI*
   • Neural Networks, CNNs, RNNs
   • TensorFlow, PyTorch, Keras
   • NLP, Computer Vision

🔸 *Data Science*
   • Data Preprocessing & EDA
   • Feature Engineering
   • Data Visualization

🔸 *Mathematics*
   • Linear Algebra, Calculus
   • Optimization, Matrix Operations

🔸 *Programming*
   • Python (Pandas, NumPy, Scikit-learn)
   • R Programming, SQL

Ask me anything from these topics! 😊🚀
"""
    bot.reply_to(message, topics_text)

def check_gratitude(text):
    """Check if user is saying thank you"""
    gratitude_words = ['thank', 'thanks', 'thx', 'appreciate', 'grateful', 'awesome', 'great', 'perfect', 'excellent']
    return any(word in text.lower() for word in gratitude_words)

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
🚫 *Hi there!* I'm StatFusionAI, specialized only in:

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
            # Send markdown response directly
            markdown_response = response.text
            
            # Check if response is too long (keep under 200 words approximately)
            if len(markdown_response) > 1500:  # Approximate 200 words
                markdown_response = markdown_response[:1500] + "... \n\n_Ask for more details if needed!_ 🤖"
            
            bot.reply_to(message, markdown_response)
        else:
            bot.reply_to(message, "❌ Hey! I couldn't generate a response. Please try rephrasing your question! 😊")
            
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        error_response = """
❌ *Oops! Something went wrong!* 😅

Please try again with a simpler question.
I'm here to help with your data science queries! 🤖
"""
        bot.reply_to(message, error_response)

# Error handler
@bot.message_handler(content_types=['photo', 'document', 'audio', 'video', 'voice', 'sticker'])
def handle_media(message):
    media_response = """
📎 Hi there! 👋

I can only process text questions about:
📊 Data Science | ML | AI | Stats | Math

*For data files:*
• Describe your dataset in text
• Ask specific analysis questions
• Request code examples

For math images:
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