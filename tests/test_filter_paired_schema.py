from jsonschema import Draft202012Validator
from agent_memory.final_filter import response_schema,decode_decisions,REASONS

def test_wire_schema_matches_the_existing_per_item_policy_for_every_pair():
    schema=response_schema(['e000'])
    Draft202012Validator.check_schema(schema)
    validator=Draft202012Validator(schema)
    accepted=rejected=0
    for decision in ['keep','drop']:
        for reason in REASONS:
            value={'query_decision':'allow','items':{'e000':{'decision':decision,'reason':reason}}}
            try:decode_decisions(value,['e000']);runtime_accepts=True
            except ValueError:runtime_accepts=False
            assert validator.is_valid(value)==runtime_accepts
            if runtime_accepts:accepted+=1
            else:rejected+=1
    assert accepted==8 and rejected==8
