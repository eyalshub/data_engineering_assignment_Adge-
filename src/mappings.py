# src/mappings.py
"""
Mappings for action types and action values.
"""

# 📄 View Content
ACTION_MAPPING = {
    "view_content": [
        "view_content", "omni_view_content", "onsite_web_view_content",
        "onsite_web_app_view_content", "onsite_app_view_content",
        "offsite_conversion.fb_pixel_view_content", "onsite_conversion.view_content"
    ],
    "add_to_cart": [
        "add_to_cart", "omni_add_to_cart", "onsite_web_add_to_cart",
        "onsite_web_app_add_to_cart", "onsite_app_add_to_cart",
        "onsite_conversion.add_to_cart", "offsite_conversion.fb_pixel_add_to_cart"
    ],
    "add_payment_info": [
        "add_payment_info", "offsite_conversion.fb_pixel_add_payment_info"
    ],
    "purchase": [
        "purchase", "omni_purchase", "onsite_web_purchase",
        "onsite_web_app_purchase", "onsite_app_purchase",
        "onsite_conversion.purchase", "offsite_conversion.fb_pixel_purchase",
        "web_app_in_store_purchase", "web_in_store_purchase"
    ],
    "initiate_checkout": [
        "initiate_checkout", "omni_initiated_checkout",
        "onsite_web_initiiate_checkout", "onsite_conversion.initiate_checkout",
        "offsite_conversion.fb_pixel_initiate_checkout"
    ],
    "complete_registration": [
        "complete_registration", "omni_complete_registration",
        "offsite_conversion.fb_pixel_complete_registration",
        "offsite_complete_registration_add_meta_leads"
    ],
    "lead": [
        "lead", "onsite_web_lead", "offsite_conversion.fb_pixel_lead"
    ],
    "landing_page_view": [
        "landing_page_view", "omni_landing_page_view"
    ],
    "post_engagement": [
        "post_engagement", "page_engagement", "post_reaction",
        "like", "comment", "post_interaction_gross", "post"
    ],
    "video_view": ["video_view"],
    "add_to_wishlist": [
        "onsite_conversion.add_to_wishlist", "omni_add_to_wishlist"
    ],
    "link_click": ["link_click"],
    "app_site_visit": ["app_site_visit"],
    "messaging_engagement": [
        "onsite_conversion.messaging_block",
        "onsite_conversion.messaging_conversation_replied_7d",
        "onsite_conversion.messaging_conversation_started_7d",
        "onsite_conversion.messaging_first_reply",
        "onsite_conversion.messaging_user_depth_2_message_send",
        "onsite_conversion.messaging_user_depth_3_message_send",
        "onsite_conversion.messaging_user_depth_5_message_send",
        "onsite_conversion.total_messaging_connection"
    ],
    "post_save": ["onsite_conversion.post_save"],
    "custom_event": [
        "offsite_conversion.custom.1039064486809880",
        "offsite_conversion.custom.1045666366274657",
        "offsite_conversion.custom.1076024029859433",
        "offsite_conversion.custom.1542671479789134",
        "offsite_conversion.custom.1727566274250500",
        "offsite_conversion.custom.1755919584614457",
        "offsite_conversion.custom.304845105021387",
        "offsite_conversion.custom.4898274876895160",
        "offsite_conversion.custom.546776554035556",
        "offsite_conversion.fb_pixel_custom"
    ]
}

ACTION_VALUES_MAPPING = {
    "view_content": [
        "view_content", "omni_view_content", "onsite_web_view_content",
        "onsite_web_app_view_content", "offsite_conversion.fb_pixel_view_content"
    ],
    "add_to_cart": [
        "add_to_cart", "omni_add_to_cart", "onsite_web_add_to_cart",
        "onsite_web_app_add_to_cart", "offsite_conversion.fb_pixel_add_to_cart"
    ],
    "add_payment_info": [
        "add_payment_info", "offsite_conversion.fb_pixel_add_payment_info"
    ],
    "purchase": [
        "purchase", "omni_purchase", "onsite_app_purchase", "onsite_web_purchase",
        "onsite_web_app_purchase", "onsite_conversion.purchase",
        "offsite_conversion.fb_pixel_purchase",
        "web_app_in_store_purchase", "web_in_store_purchase"
    ],
    "initiate_checkout": [
        "initiate_checkout", "omni_initiated_checkout",
        "onsite_web_initiate_checkout", "offsite_conversion.fb_pixel_initiate_checkout"
    ],
    "custom_event": [
        "offsite_conversion.custom.1039064486809880",
        "offsite_conversion.custom.1045666366274657",
        "offsite_conversion.custom.1076024029859433",
        "offsite_conversion.custom.1542671479789134",
        "offsite_conversion.custom.1727566274250500",
        "offsite_conversion.custom.1755919584614457",
        "offsite_conversion.custom.304845105021387",
        "offsite_conversion.custom.4898274876895160",
        "offsite_conversion.custom.546776554035556",
        "offsite_conversion.fb_pixel_custom"
    ]
}
