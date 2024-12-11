from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
from pydantic import EmailStr, BaseModel
from typing import List

from config import(
    MAIL_USERNAME,
    MAIL_PASSWORD,
    MAIL_FROM,
    MAIL_SERVER,
    MAIL_FROM_NAME,
)

class EmailServiceConfig(BaseModel):
    """Configuration model for email service settings"""
    MAIL_USERNAME: str
    MAIL_PASSWORD: str
    MAIL_FROM: EmailStr
    MAIL_PORT: int
    MAIL_SERVER: str
    MAIL_FROM_NAME: str
    MAIL_STARTTLS: bool  # Required field
    MAIL_SSL_TLS: bool   # Required field
    USE_CREDENTIALS: bool
    VALIDATE_CERTS: bool


class EmailService:
    def __init__(self, config: EmailServiceConfig):
        self.conf = ConnectionConfig(
            MAIL_USERNAME=config.MAIL_USERNAME,
            MAIL_PASSWORD=config.MAIL_PASSWORD,
            MAIL_FROM=config.MAIL_FROM,
            MAIL_PORT=config.MAIL_PORT,
            MAIL_SERVER=config.MAIL_SERVER,
            MAIL_FROM_NAME=config.MAIL_FROM_NAME,
            MAIL_STARTTLS=config.MAIL_STARTTLS,  # Using correct field
            MAIL_SSL_TLS=config.MAIL_SSL_TLS,    # Using correct field
            USE_CREDENTIALS=config.USE_CREDENTIALS,
            VALIDATE_CERTS=config.VALIDATE_CERTS
        )
        self.fast_mail = FastMail(self.conf)

    async def send_email(
        self, 
        subject: str, 
        recipients: List[EmailStr], 
        body: str, 
        subtype: str = "plain"
    ):
        message = MessageSchema(
            subject=subject,
            recipients=recipients,
            body=body,
            subtype=subtype
        )
        await self.fast_mail.send_message(message)

def get_email_service():
    config = EmailServiceConfig(
    MAIL_USERNAME=MAIL_USERNAME,
    MAIL_PASSWORD=MAIL_PASSWORD,
    MAIL_FROM=(MAIL_FROM),
    MAIL_PORT=587,
    MAIL_SERVER=MAIL_SERVER,
    MAIL_FROM_NAME=MAIL_FROM_NAME,
    MAIL_STARTTLS=True,
    MAIL_SSL_TLS=False,
    USE_CREDENTIALS=True,
    VALIDATE_CERTS=True
)
    return EmailService(config)