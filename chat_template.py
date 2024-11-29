css = '''
<style>
.chat-message {
    padding: 1.5rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    display: flex;
    background: rgba(255, 255, 255, 0); /* Fully transparent background */
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.18);
}
.chat-message .avatar {
    width: 20%;
    display: flex;
    align-items: center;
    justify-content: center;
}
.chat-message .avatar img {
    width:  90px;
    height: 90px;
    border-radius: 50%; /* Ensures circular avatars */
    object-fit: cover;
}
.chat-message .message {
    width: 80%;
    padding: 0 1.5rem;
    color: #fff;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/55vnm8z/deep-learning.png">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.ibb.co/02tpCgC/hacker.png">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''