//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "stdafx.h"
#include "CNTKLibrary.h"
#include "PrimitiveFunction.h"

namespace CNTK
{
    class BlockFunction final : public PrimitiveFunction
    {
    public:
        BlockFunction(FunctionPtr&& composite, const std::vector<std::pair<Variable, Variable>>& argumentsMap, const std::wstring& blockOpName, Dictionary&& attributes, const std::wstring& blockName = L"", const std::wstring& uid = GenerateUid(PrimitiveOpType::Block))
            : PrimitiveFunction(DetermineInputs(composite, argumentsMap, blockName), DetermineOutputs(composite, blockName), std::move(attributes), blockName, uid),
            m_composite(composite), m_blockOpName(blockOpName)
        {
            auto updatedOutputs = GetOutputVariables(true);
            auto currentOutputs = Outputs();
            for (size_t i = 0; i < currentOutputs.size(); ++i)
            {
                auto newOutputVar = updatedOutputs[i];
                auto currentOutputVar = currentOutputs[i];
                Function::ValidateOrUpdateOutput(currentOutputVar, newOutputVar, true);
                currentOutputVar.m_dataFields->m_name = newOutputVar.Name();
            }

            auto compositeOutputs = composite->Outputs();
            for (size_t i = 0; i < currentOutputs.size(); ++i)
                currentOutputs[i].m_dataFields->m_blockFunctionVariableMapping = compositeOutputs[i];
        }

        virtual const std::wstring& OpName() const override { return m_blockOpName; }

        const FunctionPtr& Composite() const { return m_composite; }

        // Mapping from each argument of the composite underlying the block to the corresponding Variable it is mapped to
        std::vector<std::pair<Variable, Variable>> CompositeArgumentsMap() const
        {
            std::unordered_map<Variable, Variable> argumentsMappingAsMap;
            auto arguments = m_composite->Arguments();
            for (auto argument : arguments)
            {
                if (argument.BlockFunctionVariableMapping() == Variable())
                    LogicError("BlockFunction (%S) with OpName (%S) does not have a mapping for argument (%S)", Name().c_str(), OpName().c_str(), argument.Name().c_str());

                argumentsMappingAsMap[argument] = argument.BlockFunctionVariableMapping();
            }

            std::vector<std::pair<Variable, Variable>> argumentsMap;
            auto blockInputs = Inputs();
            for (auto blockInput : blockInputs)
            {
                auto iter = std::find_if(argumentsMappingAsMap.begin(), argumentsMappingAsMap.end(), [&blockInput](const std::pair<Variable, Variable>& entry) {return entry.second == blockInput; });
                if (iter != argumentsMappingAsMap.end())
                    argumentsMap.push_back({iter->first, iter->second});
            }

            return argumentsMap;
        }

        // Mapping from each output of the block to the corresponding  output of underlying composite
        std::unordered_map<Variable, Variable> CompositeOutputsMap() const
        {
            std::unordered_map<Variable, Variable> outputsMap;
            auto outputs = Outputs();
            for (auto output : outputs)
            {
                if (output.BlockFunctionVariableMapping() == Variable())
                    LogicError("BlockFunction (%S) with OpName (%S) does not have a mapping for output (%S)", Name().c_str(), OpName().c_str(), output.Name().c_str());

                outputsMap[output] = output.BlockFunctionVariableMapping();
            }

            return outputsMap;
        }

    protected:
        virtual void OnPlaceholdersReplaced(const std::unordered_map<Variable, Variable>& placeholderReplacements,
                                            std::unordered_set<Variable>& replacedPlaceholders) override
        {
            // Substitute any placeholder replacements in the arguments map
            auto arguments = m_composite->Arguments();
            for (auto argument : arguments)
            {
                if (replacedPlaceholders.find(argument.BlockFunctionVariableMapping()) != replacedPlaceholders.end())
                    argument.m_dataFields->m_blockFunctionVariableMapping = placeholderReplacements.at(argument.BlockFunctionVariableMapping());
            }
        }

    private:
        static std::vector<Variable> DetermineInputs(const FunctionPtr& composite, const std::vector<std::pair<Variable, Variable>>& argumentsMap, const std::wstring& blockName)
        {
            std::unordered_map<Variable, Variable> argumentsMappingAsMap;
            for (auto argumentMapping : argumentsMap)
            {
                auto wasInserted = argumentsMappingAsMap.insert(argumentMapping).second;
                if (!wasInserted)
                    InvalidArgument("CNTK::AsBlock: Multiple mappings provided for the argument (%S) of the block composite", argumentMapping.first.Name().c_str());
            }

            std::vector<Variable> blockFunctionInputs;
            auto compositeInputs = composite->Inputs();
            std::vector<Variable> unmappedArguments;
            for (auto compositeInput : compositeInputs)
            {
                assert(!compositeInput.IsOutput());

                if (compositeInput.IsConstant() || compositeInput.IsParameter())
                    blockFunctionInputs.push_back(compositeInput);
                else
                {
                    if (!compositeInput.IsPlaceholder())
                    {
                        InvalidArgument("The composite implementing block (%S) has an argument (%S) which is not a placeholder. "
                            "All arguments of the composite underlying a block must be placeholders",
                            blockName.c_str(), compositeInput.Name().c_str());
                    }

                    // Verify that a mapping was provided for each argument of the composite
                    if (argumentsMappingAsMap.find(compositeInput) == argumentsMappingAsMap.end())
                        unmappedArguments.push_back(compositeInput);
                }
            }

            if (!unmappedArguments.empty())
            {
                auto unmappedArgumentsNames = NamedListString(unmappedArguments);
                InvalidArgument("%d arguments (%S) of the underlying composite Function of block (%S) have not been mapped when encapsulating the composite as a block", (int)unmappedArguments.size(), unmappedArgumentsNames.c_str(), blockName.c_str());
            }

            // We now append the mapped arguments of the composite to the block inputs in the order of the map
            // instead of the original order they appear in the composite itself
            for (auto argumentMapping : argumentsMap)
            {
                argumentMapping.first.m_dataFields->m_blockFunctionVariableMapping = argumentMapping.second;
                blockFunctionInputs.push_back(argumentMapping.second);
            }

            return blockFunctionInputs;
        }

        virtual std::vector<Variable> GetOutputVariables(bool inferDimensions) override
        {
            // We determine the outputs by replacing the arguments of the composite with new placeholders with updated 
            // shape etc. information matching the corresponding mapped input
            auto currentArguments = m_composite->Arguments();
            std::unordered_map<Variable, Variable> replacementMap;
            for (auto currentArgument : currentArguments)
            {
                auto currentArgumentMapping = currentArgument.BlockFunctionVariableMapping();
                auto newArgument = PlaceholderVariable(currentArgumentMapping.Shape(), currentArgumentMapping.GetDataType(), currentArgumentMapping.Name(), currentArgumentMapping.DynamicAxes());
                newArgument.m_dataFields->m_blockFunctionVariableMapping = currentArgumentMapping;

                replacementMap.insert({ currentArgument, newArgument });
            }

            m_composite->ReplacePlaceholders(replacementMap);

            // Substitute any placeholder replacements in the outputs map
            auto outputs = Outputs();
            for (auto output : outputs)
            {
                if (replacementMap.find(output.BlockFunctionVariableMapping()) != replacementMap.end())
                    output.m_dataFields->m_blockFunctionVariableMapping = replacementMap.at(output.BlockFunctionVariableMapping());
            }

            std::vector<Variable> blockFunctionOutputs;
            auto compositeOutputs = m_composite->Outputs();
            for (auto compositeOutput : compositeOutputs)
                blockFunctionOutputs.push_back(OutputVariable(compositeOutput.Shape(), compositeOutput.GetDataType(), compositeOutput.DynamicAxes(), Name()));

            return blockFunctionOutputs;
        }

        static std::vector<Variable> DetermineOutputs(const FunctionPtr& composite, const std::wstring& blockName)
        {
            std::vector<Variable> blockFunctionOutputs;
            auto compositeOutputs = composite->Outputs();
            for (auto compositeOutput : compositeOutputs)
            {
                auto output = OutputVariable(compositeOutput.Shape(), compositeOutput.GetDataType(), compositeOutput.DynamicAxes(), blockName);
                output.m_dataFields->m_blockFunctionVariableMapping = compositeOutput;

                blockFunctionOutputs.push_back(output);
            }

            return blockFunctionOutputs;
        }

    private:
        FunctionPtr m_composite;
        std::wstring m_blockOpName;

        // Increasing s_serializationVersion every time we add more ops allows us to print 
        // a more meaningful message when trying to load a new model with a stale binary. 
        static const size_t s_serializationVersion = 1;
    };
}
