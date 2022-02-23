from typing import Optional, Union

from df_engine.core.keywords import TRANSITIONS, RESPONSE
from df_engine.core import Context, Actor
import df_engine.labels as lbl
import df_engine.conditions as cnd

import re

plot = {
    "global_flow": {
        "start_node": {
            RESPONSE: "",
            TRANSITIONS: {
                ("nlp_flow", "node_1"): cnd.regexp(r"talk about NLP", re.I),
                ("greeting_flow", "node_1"): cnd.regexp(r"hi|hello|hey", re.I),
                "fallback_node": cnd.true()
            }
        },
        "fallback_node": {
            RESPONSE: "Oops",
            TRANSITIONS: {
                ("nlp_flow", "node_1"): cnd.regexp(r"talk about NLP", re.I),
                ("greeting_flow", "node_1"): cnd.regexp(r"hi|hello|hey", re.I),
                lbl.previous(): cnd.regexp(r"previous", re.I),
                lbl.repeat(): cnd.true()
            }
        }
    },
    "greeting_flow": {
        "node_1": {
            RESPONSE: "Hi, how are you?",
            TRANSITIONS: {
                lbl.to_fallback(0.1): cnd.true(),
                "node_2": cnd.regexp(r"how are you", re.I)
            }
        },
        "node_2": {
            RESPONSE: "Great! What would you like to talk about?",
            TRANSITIONS: {
                lbl.to_fallback(0.1): cnd.true(),  # == ("global_flow", "fallback_node", 0.1)
                lbl.forward(0.5): cnd.regexp(r"talk about", re.I),  # == ("greeting_flow", "node_3", 0.5)
                ("nlp_flow", "node_1"): cnd.regexp(r"talk about NLP", re.I),
                lbl.previous(): cnd.regexp(r"previous", re.I)
            }
        },
        "node_3": {
            RESPONSE: "Sorry, I can not talk about music now.",
            TRANSITIONS: {
                lbl.forward(): cnd.regexp(r"bye")
            }
        },
        "node_4": {
            RESPONSE: "understandable, have a nice day!",
            TRANSITIONS: {
                "node_1": cnd.regexp(r"hi|hello|hey", re.I),
                lbl.to_fallback(): cnd.true()
            }
        }
    },
    "nlp_flow": {
        "node_1": {
            RESPONSE: "I like to solve problems of controlled text generation. Is it interesting to you?",
            TRANSITIONS: {
                lbl.forward(): cnd.regexp(r"yes|yep|go|ok", re.I),
                ("greeting_flow", "node_4"): cnd.regexp(r"no|nope", re.I),
                lbl.to_fallback(): cnd.true()
            }
        },
        "node_2": {
            RESPONSE: "Would you like me to tell about it?",
            TRANSITIONS: {
                lbl.forward(): cnd.regexp(r"yes|yep|go|ok", re.I),
                ("greeting_flow", "node_4"): cnd.regexp(r"no|nope", re.I),
                lbl.to_fallback(): cnd.true()
            }
        },
        "node_3": {
            RESPONSE: "The base model for this task id DialoGPT - GPT-2 based model developed by Microsoft.",
            TRANSITIONS: {
                lbl.forward(): cnd.regexp(r"next", re.I),
                lbl.repeat(): cnd.regexp(r"repeat", re.I),
                lbl.to_fallback(): cnd.true()
            }
        },
        "node_4": {
            RESPONSE: "There are different approaches to accomplish this task, including prompt tuning and inverse "
                      "prompting.",
            TRANSITIONS: {
                lbl.forward(): cnd.regexp(r"next", re.I),
                lbl.repeat(): cnd.regexp(r"repeat", re.I),
                lbl.backward(): cnd.regexp(r"back", re.I),
                lbl.to_fallback(): cnd.true()
            }
        },
        "node_5": {
            RESPONSE: "You can also use Plug'n'Play model as a simpler method.",
            TRANSITIONS: {
                lbl.forward(): cnd.regexp(r"next", re.I),
                lbl.repeat(): cnd.regexp(r"repeat", re.I),
                lbl.backward(): cnd.regexp(r"back", re.I),
                lbl.to_fallback(): cnd.true()
            }
        },
        "node_6": {
            RESPONSE: "DailyDialog - a typical dataset on which DialoGPT is trained.",
            TRANSITIONS: {
                lbl.forward(): cnd.regexp(r"next", re.I),
                lbl.repeat(): cnd.regexp(r"repeat", re.I),
                lbl.backward(): cnd.regexp(r"back", re.I),
                lbl.to_fallback(): cnd.true()
            }
        },
        "node_7": {
            RESPONSE: "Woah! That's plenty of information. I think, it's time for you to dive into details)",
            TRANSITIONS: {
                ("greeting_flow", "node_2", 1.0): cnd.regexp(r"next", re.I),
                ("greeting_flow", "node_4", 2.0): cnd.regexp(r"next time", re.I),
                lbl.to_fallback(): cnd.true()
            }
        }
    }
}

actor = Actor(plot, start_label=("global_flow", "start_node"), fallback_label=("global_flow", "fallback_node"))


def turn_handler(
    in_request: str, ctx: Union[Context, str, dict], actor: Actor, true_out_response: Optional[str] = None
):
    ctx = Context.cast(ctx)
    ctx.add_request(in_request)
    ctx = actor(ctx)
    out_response = ctx.last_response
    return out_response, ctx


def run_interactive_mode(actor):
    ctx = {}
    while True:
        in_request = input("You: ")
        out_response, ctx = turn_handler(in_request, ctx, actor)
        print(f"Bot: {out_response}")


if __name__ == "__main__":
    run_interactive_mode(actor)
